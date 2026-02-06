from __future__ import annotations

import os
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from training import callbacks
from training.checkpointing import CheckpointManager
from training.metrics import compute_metrics
from training.loss import compute_total_loss, LOSS_NAMES
from training.optim import build_optimizer, build_scheduler, get_amp_dtype
from training.wandb_utils import log_metrics
from models.base import GenerativeModel


def resume_from_ckpt(
    resume_path: str,
    model: GenerativeModel,
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

    model.load_state_dict_components(ckpt)

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
    model: GenerativeModel,
    train_loader,
    device: torch.device,
    optimizer,
    scaler,
    amp_dtype: torch.dtype,
    use_amp: bool,
    grad_clip,
    sanity_steps: int,
):
    if not sanity_steps or sanity_steps <= 0:
        return

    model.train()
    it = iter(train_loader)
    for _ in range(int(sanity_steps)):
        batch = next(it)
        x = batch["input"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)
        x, y, m = callbacks.enforce_channels_last(x, y, m, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            preds = model(x)
            loss_dict = compute_total_loss(
                preds=preds,
                target=y,
                mask=m,
            )
            loss = loss_dict[LOSS_NAMES["total"]]

        if not torch.isfinite(loss):
            raise RuntimeError("Sanity check failed: non-finite loss")

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
    print(f"✅ Sanity check passed ({sanity_steps} step(s))")


def train_model(
    model: GenerativeModel,
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
    cfg=None,
    run_dir: Path = Path("."),
    shutdown_flag_path: str | None = None,
):
    if cfg is None:
        raise ValueError("cfg must be provided")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    amp_dtype = get_amp_dtype(cfg)

    model.to(device)
    if cfg.perf.channels_last:
        if str(cfg.data.num_adjacent_slices).lower() == "3d" and hasattr(torch, "channels_last_3d"):
            model = model.to(memory_format=torch.channels_last_3d)
        else:
            model = model.to(memory_format=torch.channels_last)

    optimizer = build_optimizer(model.parameters(), device, lr)
    scheduler = build_scheduler(optimizer, scheduler_patience)
    ckpt_mgr = CheckpointManager(
        run_dir / cfg.checkpoint.ckpt_dir,
        monitor="val_loss",
        mode="min",
        filename=cfg.checkpoint.ckpt_filename,
    )

    scaler_enabled = (use_amp and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler(enabled=scaler_enabled)

    if resume_path is None:
        print("🆕 Starting fresh training")
    else:
        start_epoch = resume_from_ckpt(
            resume_path,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
            use_amp,
            num_epochs,
            start_epoch,
        )

    perform_sanity_check(
        model=model,
        train_loader=train_loader,
        device=device,
        optimizer=optimizer,
        scaler=scaler,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
        grad_clip=grad_clip,
        sanity_steps=sanity_steps,
    )

    history = {"train_loss": [], "val_loss": [], "SSIM": [], "PSNR": [], "Dice_Score": []}
    end_epoch = start_epoch + num_epochs - 1
    for epoch in tqdm(range(start_epoch, end_epoch + 1), desc="Epochs", unit="epoch"):
        print(f"\n🧾 Epoch {epoch}/{end_epoch}")
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = batch["input"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            x, y, m = callbacks.enforce_channels_last(x, y, m, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                preds = model(x)
                loss_dict = compute_total_loss(
                    preds=preds,
                    target=y,
                    mask=m,
                )
                loss = loss_dict[LOSS_NAMES["total"]]

            if not torch.isfinite(loss):
                print("⚠️ Non-finite loss encountered; skipping step")
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            train_loss += float(loss.item())
            if not callbacks.SHUTDOWN_REQUESTED and shutdown_flag_path and os.path.exists(shutdown_flag_path):
                print("⏹️ Shutdown flag detected. Stopping training after current epoch.")
                callbacks.SHUTDOWN_REQUESTED = True

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss, val_loss_sem, ssim_sum, psnr_sum = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                m = batch["mask"].to(device, non_blocking=True)
                x, y, m = callbacks.enforce_channels_last(x, y, m, device)

                preds = model(x)

                loss_dict = compute_total_loss(
                    preds=preds,
                    target=y,
                    mask=m,
                )
                loss = loss_dict[LOSS_NAMES["total"]]
                sem = loss_dict[LOSS_NAMES["semantic"]]

                val_loss += float(loss.item())
                val_loss_sem += float(sem.item())
                s, p = compute_metrics(preds, y)
                ssim_sum += s
                psnr_sum += p

                if not callbacks.SHUTDOWN_REQUESTED and shutdown_flag_path and os.path.exists(shutdown_flag_path):
                    print("⏹️ Shutdown flag detected. Stopping training after current epoch.")
                    callbacks.SHUTDOWN_REQUESTED = True

        val_loss /= max(1, len(val_loader))
        sem_loss_avg = val_loss_sem / max(1, len(val_loader))
        dice_score_avg = 1 - sem_loss_avg
        ssim_avg = ssim_sum / max(1, len(val_loader))
        psnr_avg = psnr_sum / max(1, len(val_loader))

        val_metrics = {"val_loss": val_loss, "dice_score": dice_score_avg, "SSIM": ssim_avg, "PSNR": psnr_avg}
        scheduler.step(val_loss)
        ckpt_mgr.update(epoch, model, optimizer, scheduler, scaler, val_metrics)

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
            log_metrics({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "SSIM": ssim_avg,
                "PSNR": psnr_avg,
                "Dice_Score": dice_score_avg,
                "lr": optimizer.param_groups[0]["lr"],
            })

        if callbacks.SHUTDOWN_REQUESTED:
            print(f"[shutdown] Stopping at epoch {epoch} as requested.")
            if run_dir is not None:
                ckpt_mgr.save_epoch(run_dir / f"epoch_{epoch}.pt", epoch, model, optimizer, scheduler, scaler)
                print(f"Last checkpoint saved at epoch {epoch}.")
            break

    print("\n🏁 Training complete!")
    ckpt_mgr.load_best(model, optimizer, device)
    return model, history
