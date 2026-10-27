import os
import json
import argparse
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
import io
import time
import logging
from PIL import Image
from typing import Any, cast

import wandb

# Import default config
from config import cfg as default_cfg

# Model architecture (moved out of this file)
from UNET_model import build_resunet2d, build_resunet3d
import train_utils as tu
# from segmentation_module.old.segmentor import SemanticConsistencyModule

from dataset_class import MRCTDataset, build_dataloaders
from UNET_Loss import LossModule
# Shutdown flag (will be set properly in __main__)
SHUTDOWN_FLAG_PATH = None
# Reproducibility + device helpers
def seed_everything(seed: int = 42, cfg: dict | None = None):
    if cfg is None:
        cfg = default_cfg
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism vs speed (HPC training typically prefers speed)
    torch.backends.cudnn.deterministic = bool(cfg.get("deterministic", False))
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))

def _select_device():
    """Select GPU device (rank-aware) or CPU."""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if not (0 <= local_rank < torch.cuda.device_count()):
            local_rank = 0
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


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
    logger = None,
):
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info(f"🔁 Resuming from checkpoint: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        logger.info("✅ Optimizer state loaded.")
    else:
        logger.warning("⚠️ Optimizer state not found — reinitialized.")

    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "scaler" in ckpt and use_amp:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt.get("epoch", 0) + 1
    if "val_metrics" in ckpt:
        logger.info(f"📊 Previous validation metrics: {ckpt['val_metrics']}")

    logger.info(f"Resuming training from epoch {start_epoch}/{num_epochs}")
    return start_epoch


def perform_sanity_check(
    encoder: nn.Module,
    decoder: nn.Module,
    train_loader,
    device: torch.device,
    optimizer,
    scaler,
    # criterion,
    loss_module: LossModule,
    amp_dtype: torch.dtype,
    use_amp: bool,
    # lambda_ssim: float,
    # ssim_down: int,
    grad_clip,
    sanity_steps: int,
    # run_model,
    logger = None,
):
    if logger is None:
        logger = logging.getLogger(__name__)
    if not sanity_steps or sanity_steps <= 0:
        return

    encoder.train()
    decoder.train()
    it = iter(train_loader)
    for _ in range(int(sanity_steps)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        x = batch["input"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)
        x, y, m = tu.enforce_channels_last(x, y, m, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            preds = decoder(encoder(x))
            total_loss, _, _, _ = loss_module(preds, y, m)
        if not torch.isfinite(total_loss):
            raise RuntimeError("Sanity check failed: non-finite loss")

        if scaler.is_enabled():
            scaler.scale(total_loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), grad_clip)
            optimizer.step()
    logger.info(f"✅ Sanity check passed ({sanity_steps} step(s))")

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
    loss_module: LossModule | None = None,
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
    if loss_module is None:
        loss_module = LossModule(device)

    amp_dtype = tu.get_amp_dtype(cfg)
    is_3d = str(cfg.get("num_adjacent_slices", 0)).lower() == "3d"

    encoder.to(device)
    decoder.to(device)
    if cfg.get("channels_last", False):
        if is_3d and hasattr(torch, "channels_last_3d"):
            encoder = cast(Any, encoder).to(memory_format=torch.channels_last_3d)
            decoder = cast(Any, decoder).to(memory_format=torch.channels_last_3d)
        else:
            encoder = cast(Any, encoder).to(memory_format=torch.channels_last)
            decoder = cast(Any, decoder).to(memory_format=torch.channels_last)

    # criterion = nn.L1Loss()
    # semantic_criterion = SemanticConsistencyModule(device)
    # lambda_sem = 0.1 # Weight for semantic loss
    optimizer = tu.build_optimizer(encoder, decoder, device, lr)
    scheduler = tu.build_scheduler(optimizer, scheduler_patience)
    ckpt_mgr = tu.CheckpointManager(run_dir / cfg["ckpt_dir"], monitor="val_loss", mode="min", filename=cfg["ckpt_filename"])

    scaler_enabled = (use_amp and amp_dtype == torch.float16)
    grad_scaler_cls = getattr(getattr(torch, "amp", None), "GradScaler", torch.cuda.amp.GradScaler)
    scaler = grad_scaler_cls(enabled=scaler_enabled)
    
    logger = logging.getLogger(__name__)
    if resume_path is None:
        logger.info("🆕 Starting fresh training")
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
            logger=logger,
        )

    perform_sanity_check(
        encoder=encoder,
        decoder=decoder,
        train_loader=train_loader,
        device=device,
        optimizer=optimizer,
        scaler=scaler,
        loss_module=loss_module,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
        # lambda_ssim=lambda_ssim,
        # ssim_down=ssim_down,
        grad_clip=grad_clip,
        sanity_steps=sanity_steps,
        logger=logger,
    )

    def _check_shutdown_requested() -> bool:
        if not tu.SHUTDOWN_REQUESTED and shutdown_flag_path and os.path.exists(shutdown_flag_path):
            logger.warning("⏹️ Shutdown flag detected. Stopping training after current epoch.")
            tu.SHUTDOWN_REQUESTED = True
        return tu.SHUTDOWN_REQUESTED

    def format_time(seconds):
        """Convert seconds to human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    logger = logging.getLogger(__name__)
    history = {"train_loss": [], "val_loss": [], "SSIM": [], "PSNR": [], "Dice_Score": []}
    end_epoch = start_epoch + num_epochs - 1
    training_start_time = time.time()
    epoch_times = []
    
    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch}/{end_epoch} started")
        encoder.train()
        decoder.train()
        train_loss = 0.0
        train_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            x = batch["input"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            x, y, m = tu.enforce_channels_last(x, y, m, device)



            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                preds = decoder(encoder(x))
                total_loss, _, _, _ = loss_module(preds, y, m)

            if not torch.isfinite(total_loss):
                logger.warning("⚠️ Non-finite loss encountered; skipping step")
                continue

            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), grad_clip)
                optimizer.step()

            train_loss += float(total_loss.item())
            train_steps += 1
            _check_shutdown_requested()
            
            # Log progress every 10% of batches
            if (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
                logger.info(f"Epoch {epoch}/{end_epoch} - Train batch {batch_idx+1}/{len(train_loader)} - Loss: {total_loss.item():.4f}")

        train_loss /= max(1, train_steps)

        encoder.eval()
        decoder.eval()
        val_loss, ssim_sum, psnr_sum, seg_loss_sum = 0.0, 0.0, 0.0, 0.0
        val_steps = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                x = batch["input"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                m = batch["mask"].to(device, non_blocking=True)
                x, y, m = tu.enforce_channels_last(x, y, m, device)
                preds = decoder(encoder(x))

                total_loss, _, _, seg_loss_val = loss_module(preds, y, m)

                val_loss += float(total_loss.item())
                seg_loss_sum += seg_loss_val
                s, p = loss_module.compute_val_metrics(preds, y)
                ssim_sum += s
                psnr_sum += p
                val_steps += 1
                
                # Log progress every 10% of batches
                if (batch_idx + 1) % max(1, len(val_loader) // 10) == 0:
                    logger.info(f"Epoch {epoch}/{end_epoch} - Val batch {batch_idx+1}/{len(val_loader)} - Loss: {total_loss.item():.4f}")
                
                _check_shutdown_requested()

        val_loss /= max(1, val_steps)
        seg_loss_avg = seg_loss_sum / max(1, val_steps)
        dice_score_avg = 1.0 - seg_loss_avg
        ssim_avg = ssim_sum / max(1, val_steps)
        psnr_avg = psnr_sum / max(1, val_steps)

        val_metrics = {"val_loss": val_loss, "SSIM": ssim_avg, "PSNR": psnr_avg, "Dice_Score": dice_score_avg}
        scheduler.step(val_loss)
        ckpt_mgr.update(epoch, encoder, decoder, optimizer, scheduler, scaler, val_metrics)

        # Calculate time metrics
        epoch_duration = time.time() - epoch_start_time
        epoch_times.append(epoch_duration)
        elapsed_time = time.time() - training_start_time
        
        # Estimate time remaining based on average epoch time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        epochs_remaining = end_epoch - epoch
        estimated_time_remaining = avg_epoch_time * epochs_remaining
        
        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"SSIM: {ssim_avg:.4f} | PSNR: {psnr_avg:.2f} | "
            f"Dice: {dice_score_avg:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e}"
        )
        logger.info(
            f"Time - Epoch: {format_time(epoch_duration)} | "
            f"Elapsed: {format_time(elapsed_time)} | "
            f"Remaining: {format_time(estimated_time_remaining)} (est.)"
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

            should_log_image = log_every > 0 and (epoch % log_every == 0)
            should_save_image = save_every > 0 and (epoch % save_every == 0)
            if should_log_image or should_save_image:
                fig = tu.visualize_batch(encoder, decoder, val_loader, device=device, num_samples=4)

                if should_log_image:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    image = Image.open(buf)
                    wandb.log({f"epoch_{epoch}_output": wandb.Image(image, caption=f"Epoch {epoch}")})

                if should_save_image:
                    vis_name = cfg["Visualisation_Fig"]
                    fig.savefig(str(run_dir / f"epoch_{epoch}_{vis_name}"))

                plt.close(fig)

        if tu.SHUTDOWN_REQUESTED:
            logger.info(f"[shutdown] Stopping at epoch {epoch} as requested.")
            if run_dir is not None:
                ckpt_mgr.save_epoch(run_dir / f"epoch_{epoch}.pt", epoch, encoder, decoder, optimizer, scheduler, scaler)
                logger.info(f"Last checkpoint saved at epoch {epoch}.")
            break

    # logger.info("\n🏁 Training complete!")
    # ckpt_mgr.load_best(encoder, decoder, optimizer, device)
    return encoder, decoder, history

def main(cfg: dict):
    """Main training pipeline."""
    global SHUTDOWN_FLAG_PATH
    SHUTDOWN_FLAG_PATH = os.path.join(cfg["run_dir"], "stop.flag")
    
    # =========================================================================
    # Setup & Initialization
    # =========================================================================
    run_dir = Path(cfg["run_dir"])
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Starting UNET training")
    logger.info("=" * 60)
    
    SEED = 42
    seed_everything(SEED, cfg)

    # Device selection and AMP setup
    device = _select_device()
    use_amp = (device.type == "cuda")
    logger.info(f"Device: {device} | AMP: {use_amp} | CUDA devices: {torch.cuda.device_count()}")

    # H100-friendly math (TF32 for float32 matmuls/convs)
    if device.type == "cuda" and cfg.get("allow_tf32", True):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    # Configure global torch settings
    root = cfg["root"]
    MAX_EPOCHS = cfg["epochs"]
    IS_3D = str(cfg.get("num_adjacent_slices", 0)).lower() == "3d"
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        logger.warning("⚠️ torch.set_float32_matmul_precision not available; continuing without it")

    # Setup run directory and W&B
    run_dir = Path(cfg["run_dir"])
    tu.init_wandb(cfg, SEED, use_amp, IS_3D)

    # =========================================================================
    # Dataset Preparation
    # =========================================================================
    dataset = MRCTDataset(
        root=root,
        modality=cfg["modality"],
        use_preproc_files=cfg.get("use_preproc_files", False),
        use_pruned_data=cfg.get("use_pruned_data", False),
        patch_size=cfg["patch_size"],
        anatomy=cfg.get("anatomy", None),
        num_adjacent_slices=cfg["num_adjacent_slices"],
        max_retries=20,
        foreground_threshold=cfg["foreground_threshold"],
        cache_size=cfg["cache_size"],  # per-worker LRU
    )
    logger.info(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    sample_input = sample["input"]
    sample_label = sample.get("label")
    sample_mask = sample.get("mask")
    if sample_label is None or sample_mask is None:
        raise KeyError("Dataset sample must include 'label' and 'mask' keys")
    logger.info(f"Sample shapes - input: {sample_input.shape}, label: {sample_label.shape}, mask: {sample_mask.shape}")
    if IS_3D:
        assert sample_input.dim() == 4, "Expected 3D input with shape (C, D, H, W)"
    else:
        assert sample_input.shape[0] == 1, "Expected 2D input with 1 channel"

    train_loader, val_loader, train_ds, val_ds, train_patients, val_patients = build_dataloaders(
        dataset, cfg, SEED, device
    )

    logger.info(f"Patients: {len(train_patients)} train | {len(val_patients)} val")
    logger.info(f"Samples: {len(train_ds)} train | {len(val_ds)} val")

    # =========================================================================
    # Model Building
    # =========================================================================
    if IS_3D:
        model_cfg = cfg["model3d_cfg"]
        enc, dec = build_resunet3d(model_cfg)
        logger.info(f"Using 3D model configuration: {model_cfg}")
    else:
        model_cfg = cfg["model2d_cfg"]
        enc, dec = build_resunet2d(model_cfg)
        logger.info(f"Using 2D model configuration: {model_cfg}")

    # Optional: compile models for faster inference/training
    enc, dec, compile_enabled = tu.maybe_compile_models(enc, dec, cfg, device)
    if compile_enabled:
        logger.info("✅ Models compiled successfully")
    else:
        logger.warning("⚠️ Models not compiled; running in eager mode")

    # =========================================================================
    # Training
    # =========================================================================
    lambda_ssim = cfg["lambda_ssim"]
    lambda_seg = cfg.get("lambda_seg", 0.1)
    ssim_down = int(cfg.get("ssim_downsample", 1))
    loss_module = LossModule(device, ssim_down=ssim_down, ssim_weight=lambda_ssim, seg_weight=lambda_seg)
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
        loss_module=loss_module,
        scheduler_patience=cfg["scheduler_patience"],
        grad_clip=cfg["grad_clip"],
        sanity_steps=cfg["sanity_steps"],
        resume_path=cfg["resume_path"],
        cfg=cfg,
        run_dir=run_dir,
        shutdown_flag_path=SHUTDOWN_FLAG_PATH,
    )

    # =========================================================================
    # Visualization & Cleanup
    # =========================================================================
    # Plot each training metric separately
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
        logger.info("[shutdown] Shutdown complete. Removing flag file.")
        os.remove(SHUTDOWN_FLAG_PATH)
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


def _infer_type(value):
    """Infer argparse type from a config value."""
    if value is None:
        return str
    if isinstance(value, bool):
        return str  # Handle bools specially via choices
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    return str


def _parse_value(value_str: str, default_value):
    """Parse a string value back to the appropriate type based on default."""
    if default_value is None:
        # Could be None, int, float, bool, or string
        if value_str.lower() == "none":
            return None
        if value_str.lower() in ("true", "false"):
            return value_str.lower() == "true"
        try:
            return int(value_str)
        except ValueError:
            pass
        try:
            return float(value_str)
        except ValueError:
            pass
        return value_str
    
    if isinstance(default_value, bool):
        return value_str.lower() in ("true", "1", "yes")
    if isinstance(default_value, int):
        return int(value_str)
    if isinstance(default_value, float):
        return float(value_str)
    if isinstance(default_value, (list, tuple)):
        # Try to parse as JSON first
        try:
            result = json.loads(value_str)
        except json.JSONDecodeError:
            # Fallback: split by comma and parse each element
            result = []
            for item in value_str.split(','):
                item = item.strip()
                # Try to parse as int, float, or keep as string
                try:
                    result.append(int(item))
                except ValueError:
                    try:
                        result.append(float(item))
                    except ValueError:
                        result.append(item)
        # Convert to tuple if default was tuple
        return tuple(result) if isinstance(default_value, tuple) else result
    if isinstance(default_value, dict):
        return json.loads(value_str)
    # For string defaults, still try to parse as int/bool/float if value looks like one
    # This handles polymorphic config values like num_adjacent_slices ("3d" or 0)
    if value_str.lower() in ("true", "false"):
        return value_str.lower() == "true"
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def build_argparser(cfg: dict) -> argparse.ArgumentParser:
    """Build argparser dynamically from cfg dict."""
    parser = argparse.ArgumentParser(
        description="UNET Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    for key, value in cfg.items():
        # run_dir is required, others are optional
        if key == "run_dir":
            parser.add_argument(
                f"--{key}",
                type=str,
                required=True,
                help=f"(REQUIRED) Run directory. Default from config: {value}",
            )
        elif isinstance(value, bool):
            # For bools, use store_true/store_false pattern
            parser.add_argument(
                f"--{key}",
                type=str,
                default=str(value),
                choices=["true", "false", "True", "False", "1", "0"],
                help=f"Default: {value}",
            )
        elif isinstance(value, (list, tuple, dict)):
            # For complex types, accept JSON string
            parser.add_argument(
                f"--{key}",
                type=str,
                default=json.dumps(value),
                help=f"JSON string. Default: {json.dumps(value)}",
            )
        else:
            arg_type = _infer_type(value)
            parser.add_argument(
                f"--{key}",
                type=arg_type if value is not None else str,
                default=value if value is not None else "None",
                help=f"Default: {value}",
            )
    
    # Add log_file argument
    parser.add_argument(
        "--log_file",
        type=str,
        default="train.log",
        help="Log file name (default: train.log)",
    )
    
    return parser


def update_cfg_from_args(cfg: dict, args: argparse.Namespace) -> dict:
    """Update cfg dict with parsed arguments."""
    updated_cfg = cfg.copy()
    
    for key, default_value in cfg.items():
        if hasattr(args, key):
            arg_value = getattr(args, key)
            # Parse the value back to appropriate type
            if isinstance(arg_value, str):
                updated_cfg[key] = _parse_value(arg_value, default_value)
            else:
                updated_cfg[key] = arg_value
    
    return updated_cfg


def save_config_to_run_dir(cfg: dict, run_dir: Path, logger = None):
    """Save the final config as config.py in the run directory."""
    if logger is None:
        logger = logging.getLogger(__name__)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.py"
    
    # Build the config file content
    lines = ["cfg = {\n"]
    for key, value in cfg.items():
        if isinstance(value, str):
            lines.append(f'    "{key}": {repr(value)},\n')
        else:
            lines.append(f'    "{key}": {repr(value)},\n')
    lines.append("}\n")
    
    with open(config_path, "w") as f:
        f.writelines(lines)
    
    logger.info(f"✅ Config saved to: {config_path}")
    return config_path


if __name__ == "__main__":
    # Build argparser dynamically from default config
    parser = build_argparser(default_cfg)
    args = parser.parse_args()
    
    # Update config with CLI arguments
    cfg = update_cfg_from_args(default_cfg, args)
    
    # Ensure run_dir exists and save config there
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging with both console and file handlers
    log_path = run_dir / args.log_file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),  # mode='w' to overwrite
        ],
    )
    logger = logging.getLogger(__name__)
    
    config_path = save_config_to_run_dir(cfg, run_dir, logger=logger)
    
    # Set CONFIG_PATH environment variable so other modules read from this location
    os.environ["CONFIG_PATH"] = str(config_path)
    logger.info(f"📁 CONFIG_PATH set to: {config_path}")
    
    # Update SHUTDOWN_FLAG_PATH with actual run_dir
    SHUTDOWN_FLAG_PATH = os.path.join(cfg["run_dir"], "stop.flag")
    
    # Run training
    main(cfg)
