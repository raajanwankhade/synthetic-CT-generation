import os
import json
secrets_path = 'secrets.json'
with open(secrets_path, 'r') as f:
    secrets = json.load(f)
os.environ["KAGGLE_USERNAME"] = secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = secrets["KAGGLE_KEY"]
os.environ["WANDB_API_KEY"] = secrets["wandb_API_KEY"]
# General
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import io
from PIL import Image

import wandb
from config import cfg

# Model architecture (moved out of this file)
from UNET_model import build_resunet2d, build_resunet3d
import train_utils as tu
from segmentor import SemanticConsistencyModule

# Shutdown flag
SHUTDOWN_FLAG_PATH = os.path.join(cfg["run_dir"], "stop.flag")
# Reproducibility + device helpers
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism vs speed (HPC training typically prefers speed)
    torch.backends.cudnn.deterministic = bool(cfg.get("deterministic", False))
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))

SEED = 42
seed_everything(SEED)

# -------------------------------------------------------------------------
# Device selection (rank-aware) + AMP safety
# -------------------------------------------------------------------------
def _select_device():
    """Select a valid CUDA device (if available) and set it as current.

    Fixes cases where CUDA is available but the current device id is invalid
    (common in SLURM/MPI launches when the process rank isn't mapped).
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        # Prefer per-process local rank if provided by launcher
        for env_key in ("LOCAL_RANK", "SLURM_LOCALID", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK"):
            if env_key in os.environ:
                try:
                    idx = int(os.environ[env_key])
                    idx = idx % torch.cuda.device_count()
                    torch.cuda.set_device(idx)
                    return torch.device(f"cuda:{idx}")
                except Exception:
                    break
        # Fallback to device 0
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        return torch.device("cuda:0")
    return torch.device("cpu")


device = _select_device()
use_amp = (device.type == "cuda")
print("Device:", device, "| AMP:", use_amp, "| CUDA devices:", torch.cuda.device_count())

# H100-friendly math (TF32 for float32 matmuls/convs)
if device.type == "cuda" and cfg.get("allow_tf32", True):
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
root = cfg["root"]
MAX_EPOCHS = cfg["epochs"]
IS_3D = str(cfg.get("num_adjacent_slices", 0)).lower() == "3d"
# Faster matmul on newer torch (safe to ignore if unavailable)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Training
from dataset_class import MRCTDataset, build_dataloaders

#Run dir
run_dir = Path(cfg["run_dir"])
tu.init_wandb(cfg, SEED, use_amp, IS_3D)


def resume_from_ckpt(
    resume_path: str ,
    encoder: nn.Module,
    decoder: nn.Module,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    use_amp: bool,
    num_epochs: int,
    start_epoch: int,
):
    print(f"🔁 Resuming from checkpoint: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        print("✅ Optimizer state loaded.")
    else:
        print("⚠️ Optimizer state not found — reinitialized.")

    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "scaler" in ckpt and use_amp:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt.get("epoch", 0) + 1
    if "val_metrics" in ckpt:
        print(f"📊 Previous validation metrics: {ckpt['val_metrics']}")

    print(f"Resuming training from epoch {start_epoch}/{num_epochs}")
    return start_epoch


def perform_sanity_check(
    encoder: nn.Module,
    decoder: nn.Module,
    train_loader,
    device: torch.device,
    optimizer,
    scaler,
    criterion,
    amp_dtype: torch.dtype,
    use_amp: bool,
    lambda_ssim: float,
    ssim_down: int,
    grad_clip,
    sanity_steps: int,
    run_model,
):
    if not sanity_steps or sanity_steps <= 0:
        return

    encoder.train()
    decoder.train()
    it = iter(train_loader)
    for _ in range(int(sanity_steps)):
        batch = next(it)
        x = batch["input"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)
        x, y, m = tu.enforce_channels_last(x, y,m,  device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            preds = run_model(x)
            loss_l1 = criterion(preds, y)
            loss_ssim = tu.ssim_loss(preds, y, ssim_down)
            loss = loss_l1 + lambda_ssim * loss_ssim

        if not torch.isfinite(loss):
            raise RuntimeError("Sanity check failed: non-finite loss")

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), grad_clip)
            optimizer.step()
    print(f"✅ Sanity check passed ({sanity_steps} step(s))")


def train_model(
    encoder: nn.Module,
    decoder: nn.Module,
    train_loader,
    val_loader,
    num_epochs=100,
    start_epoch=1,
    lr=1e-4,
    scheduler_patience=5,
    device: torch.device | None = None,
    wandb_log=True,
    lambda_ssim=1000.0,
    resume_path=None,
    grad_clip=None,
    sanity_steps=1,
    cfg: dict | None = None,
    run_dir: Path = Path("."),
    shutdown_flag_path: str | None = None,
):
    if cfg is None:
        raise ValueError("cfg must be provided")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    amp_dtype = tu.get_amp_dtype(cfg)

    encoder.to(device)
    decoder.to(device)
    if cfg.get("channels_last", False):
        if IS_3D and hasattr(torch, "channels_last_3d"):
            encoder = encoder.to(memory_format=torch.channels_last_3d)
            decoder = decoder.to(memory_format=torch.channels_last_3d)
        else:
            encoder = encoder.to(memory_format=torch.channels_last)
            decoder = decoder.to(memory_format=torch.channels_last)

    encoder_eager = encoder
    decoder_eager = decoder
    encoder, decoder, compile_enabled = tu.maybe_compile_models(encoder, decoder, cfg)

    def _run_model(x: torch.Tensor):
        nonlocal encoder, decoder, compile_enabled
        try:
            feats = encoder(x)
            return decoder(feats)
        except Exception as e:
            if compile_enabled:
                print(f"⚠️ torch.compile failed at runtime; falling back to eager: {e}")
                encoder = encoder_eager
                decoder = decoder_eager
                compile_enabled = False
                feats = encoder(x)
                return decoder(feats)
            raise

    criterion = nn.L1Loss()
    semantic_criterion = SemanticConsistencyModule(device)
    lambda_sem = 0.1 # Weight for semantic loss
    optimizer = tu.build_optimizer(encoder, decoder, device, lr)
    scheduler = tu.build_scheduler(optimizer, scheduler_patience)
    ckpt_mgr = tu.CheckpointManager(run_dir / cfg["ckpt_dir"], monitor="val_loss", mode="min", filename=cfg["ckpt_filename"])

    scaler_enabled = (use_amp and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler(enabled=scaler_enabled)

    ssim_down = int(cfg.get("ssim_downsample", 1))
    
    if resume_path is None:
        print("🆕 Starting fresh training")
    else:
        start_epoch = resume_from_ckpt(
            resume_path,
            encoder,
            decoder,
            optimizer,
            scheduler,
            scaler,
            device,
            use_amp,
            num_epochs,
            start_epoch,
        )

    perform_sanity_check(
        encoder=encoder,
        decoder=decoder,
        train_loader=train_loader,
        device=device,
        optimizer=optimizer,
        scaler=scaler,
        criterion=criterion,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
        lambda_ssim=lambda_ssim,
        ssim_down=ssim_down,
        grad_clip=grad_clip,
        sanity_steps=sanity_steps,
        run_model=_run_model,
    )

    history = {"train_loss": [], "val_loss": [], "SSIM": [], "PSNR": [], "Dice_Score": []}
    end_epoch = start_epoch + num_epochs - 1
    for epoch in tqdm(range(start_epoch, end_epoch + 1), desc="Progress"):
        print(f"\n🧾 Epoch {epoch}/{end_epoch}")
        encoder.train()
        decoder.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            x = batch["input"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            x, y, m = tu.enforce_channels_last(x, y, m, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                preds = _run_model(x)
                loss_l1 = criterion(preds, y)
                loss_ssim = tu.ssim_loss(preds, y, ssim_down)

                # --- NEW: Semantic Loss ---
                # Pass the generated sCT (preds) and the Ground Truth Mask (m)
                loss_sem = semantic_criterion(preds, m)

                loss = loss_l1 + lambda_ssim * loss_ssim + lambda_sem * loss_sem

            if not torch.isfinite(loss):
                print("⚠️ Non-finite loss encountered; skipping step")
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), grad_clip)
                optimizer.step()

            train_loss += float(loss.item())
            if not tu.SHUTDOWN_REQUESTED and shutdown_flag_path and os.path.exists(shutdown_flag_path):
                print("⏹️ Shutdown flag detected. Stopping training after current epoch.")
                tu.SHUTDOWN_REQUESTED = True
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= max(1, len(train_loader))

        encoder.eval()
        decoder.eval()
        val_loss, val_loss_sem, ssim_sum, psnr_sum = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                x = batch["input"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                m = batch["mask"].to(device, non_blocking=True)
                x, y, m = tu.enforce_channels_last(x, y, m, device)

                preds = _run_model(x)

                sem = semantic_criterion(preds, m)

                loss_l1 = criterion(preds, y)
                loss_ssim = tu.ssim_loss(preds, y, ssim_down)
                loss = loss_l1 + lambda_ssim * loss_ssim + lambda_sem * sem

                val_loss += float(loss.item())
                val_loss_sem += float(sem.item())
                s, p = tu.compute_metrics(preds, y)
                ssim_sum += s
                psnr_sum += p
                
                if not tu.SHUTDOWN_REQUESTED and shutdown_flag_path and os.path.exists(shutdown_flag_path):
                    print("⏹️ Shutdown flag detected. Stopping training after current epoch.")
                    tu.SHUTDOWN_REQUESTED = True

        val_loss /= max(1, len(val_loader))
        sem_loss_avg = val_loss_sem / max(1, len(val_loader))
        dice_score_avg = 1- sem_loss_avg  
        ssim_avg = ssim_sum / max(1, len(val_loader))
        psnr_avg = psnr_sum / max(1, len(val_loader))

        val_metrics = {"val_loss": val_loss, "dice_score": dice_score_avg, "SSIM": ssim_avg, "PSNR": psnr_avg}
        scheduler.step(val_loss)
        ckpt_mgr.update(epoch, encoder, decoder, optimizer, scheduler, scaler, val_metrics)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"SSIM: {ssim_avg:.4f} | PSNR: {psnr_avg:.2f} | Dice Score: {dice_score_avg:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e}"
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["SSIM"].append(ssim_avg)
        history["PSNR"].append(psnr_avg)
        history["Dice_Score"].append(dice_score_avg)

        if wandb_log:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "SSIM": ssim_avg,
                "PSNR": psnr_avg,
                "Dice_Score": dice_score_avg,
                "lr": optimizer.param_groups[0]["lr"],
            })

            log_every = int(cfg.get("log_images_every", 0))
            save_every = int(cfg.get("save_vis_every", 0))

            if log_every > 0 and (epoch % log_every == 0):
                fig = tu.visualize_batch(encoder, decoder, val_loader, device=device, num_samples=4)
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                image = Image.open(buf)
                wandb.log({f"epoch_{epoch}_output": wandb.Image(image, caption=f"Epoch {epoch}")})
                plt.close(fig)

            if save_every > 0 and (epoch % save_every == 0):
                fig = tu.visualize_batch(encoder, decoder, val_loader, device=device, num_samples=4)
                vis_name = cfg["Visualisation_Fig"]
                fig.savefig(str(run_dir / f"epoch_{epoch}_{vis_name}"))
                plt.close(fig)

        if tu.SHUTDOWN_REQUESTED:
            print(f"[shutdown] Stopping at epoch {epoch} as requested.")
            if run_dir is not None:
                ckpt_mgr.save_epoch(run_dir / f"epoch_{epoch}.pt", epoch, encoder, decoder, optimizer, scheduler, scaler)
                print(f"Last checkpoint saved at epoch {epoch}.")
            break

    print("\n🏁 Training complete!")
    ckpt_mgr.load_best(encoder, decoder, optimizer, device)
    return encoder, decoder, history

dataset = MRCTDataset(
    root=root,
    modality=cfg["modality"],
    patch_size=None if IS_3D else cfg["patch_size"],
    anatomy=cfg.get("anatomy", None),
    num_adjacent_slices=cfg["num_adjacent_slices"],
    max_retries=20,
    foreground_threshold=cfg["foreground_threshold"],
    cache_size=cfg["cache_size"],  # per-worker LRU
)
print("Dataset size:", len(dataset))
sample = dataset[0]
print(sample["input"].shape, sample["label"].shape, sample["mask"].shape)
if IS_3D:
    assert sample["input"].dim() == 4, "Expected 3D input with shape (C, D, H, W)"
else:
    assert sample["input"].shape[0] == 1, "Expected 2D input with 1 channel"

train_loader, val_loader, train_ds, val_ds, train_patients, val_patients = build_dataloaders(
    dataset, cfg, SEED, device
)

print("Patients:", len(train_patients), "train |", len(val_patients), "val")
print("Samples:", len(train_ds), "train |", len(val_ds), "val")

# Build model
if IS_3D:
    model_cfg = cfg["model3d_cfg"]
    enc, dec = build_resunet3d(model_cfg)
else:
    enc, dec = build_resunet2d(
        in_ch=1,
        out_ch=1,
        base_ch=32,
        num_stages=4,
        blocks_per_stage=2,
    )


trained_enc, trained_dec, history = train_model(
    encoder=enc,
    decoder=dec,
    train_loader=train_loader,
    val_loader=val_loader,
    start_epoch=cfg["start"],
    num_epochs=MAX_EPOCHS,
    lr=cfg["lr"],
    device=device,
    wandb_log=(cfg["wandb_mode"] != "disabled"),
    lambda_ssim=cfg["lambda_ssim"],
    scheduler_patience=cfg["scheduler_patience"],
    grad_clip=cfg["grad_clip"],
    sanity_steps=cfg["sanity_steps"],
    resume_path=cfg["resume_path"],
    cfg=cfg,
    run_dir=run_dir,
    shutdown_flag_path=SHUTDOWN_FLAG_PATH,
)

# --- Plot each training metric separately ---
epochs = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(14, 18))

# 1. Training & Validation Loss
plt.subplot(4, 1, 1)
plt.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
plt.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Composite Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# 2. SSIM
plt.subplot(4, 1, 2)
plt.plot(epochs, history["SSIM"], label="SSIM", color='tab:blue', linewidth=2)
plt.title("SSIM over Epochs")
plt.xlabel("Epoch")
plt.ylabel("SSIM")
plt.legend()
plt.grid(True, alpha=0.3)

# 3. PSNR
plt.subplot(4, 1, 3)
plt.plot(epochs, history["PSNR"], label="PSNR", color='tab:orange', linewidth=2)
plt.title("PSNR over Epochs")
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Dice Score
plt.subplot(4, 1, 4)
plt.plot(epochs, history["Dice_Score"], label="Dice Score", color='tab:green', linewidth=2)
plt.title("Dice Score over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(run_dir / cfg["Loss_Plot_name"]))

wandb.finish()
tu.visualize_batch(trained_enc, trained_dec, val_loader)

if tu.SHUTDOWN_REQUESTED:
    print("[shutdown] Shutdown complete. Removing flag file.")
    os.remove(SHUTDOWN_FLAG_PATH)
