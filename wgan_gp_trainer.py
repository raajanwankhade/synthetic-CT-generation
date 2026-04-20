import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import wandb

import train_utils as tu
from UNET_Loss import LossModule
from wgan_gp_utils import (
    _build_critic_optimizer,
    _check_shutdown_requested,
    _format_time,
    _gradient_penalty,
    _resume_from_checkpoint,
    _run_generator,
    _save_checkpoint,
    _set_requires_grad,
)


def train_wgan_gp(
    encoder: nn.Module,
    decoder: nn.Module,
    critic: nn.Module,
    train_loader,
    val_loader,
    loss_module: LossModule,
    cfg: dict,
    run_dir: Path,
    device: torch.device,
    num_epochs: int,
    start_epoch: int,
    resume_path: str | None,
    grad_clip,
    shutdown_flag_path: str | None,
) -> tuple[nn.Module, nn.Module, nn.Module, dict[str, list[float]]]:
    logger = logging.getLogger(__name__)

    use_amp = device.type == "cuda"
    amp_dtype = tu.get_amp_dtype(cfg)
    scaler_enabled = use_amp and amp_dtype == torch.float16
    grad_scaler_cls = getattr(getattr(torch, "amp", None), "GradScaler", torch.cuda.amp.GradScaler)
    scaler = grad_scaler_cls(enabled=scaler_enabled)

    wcfg = cfg.get("wgan_gp_cfg", {})
    lambda_adv = float(wcfg.get("lambda_adv", cfg.get("lambda_adv", 0.01)))
    lambda_gp = float(wcfg.get("lambda_gp", 10.0))
    n_critic = max(1, int(wcfg.get("n_critic", 5)))
    recompute_fake_each_critic_step = bool(wcfg.get("recompute_fake_each_critic_step", False))

    gen_lr = float(wcfg.get("gen_lr", cfg.get("lr", 1e-4)))
    disc_lr = float(wcfg.get("disc_lr", gen_lr))
    disc_betas_raw = wcfg.get("disc_betas", (0.0, 0.9))
    if not isinstance(disc_betas_raw, (list, tuple)) or len(disc_betas_raw) != 2:
        raise ValueError("wgan_gp_cfg['disc_betas'] must be a list/tuple with exactly two values")
    disc_betas = (float(disc_betas_raw[0]), float(disc_betas_raw[1]))

    critic_grad_clip = wcfg.get("critic_grad_clip", None)
    if critic_grad_clip is not None:
        critic_grad_clip = float(critic_grad_clip)

    gen_scheduler_patience = int(wcfg.get("gen_scheduler_patience", cfg.get("scheduler_patience", 10)))
    disc_scheduler_patience = int(wcfg.get("disc_scheduler_patience", cfg.get("scheduler_patience", 10)))

    gen_optimizer = tu.build_optimizer(encoder, decoder, device, gen_lr)
    critic_optimizer = _build_critic_optimizer(critic, device, disc_lr, disc_betas)
    gen_params = list(encoder.parameters()) + list(decoder.parameters())

    gen_scheduler = tu.build_scheduler(gen_optimizer, gen_scheduler_patience)
    critic_scheduler = tu.build_scheduler(critic_optimizer, disc_scheduler_patience)

    ckpt_dir = run_dir / cfg.get("ckpt_dir", "checkpoints")
    best_ckpt_name = wcfg.get("best_ckpt_filename", "best_wgan_gp.pth")
    best_ckpt_path = ckpt_dir / best_ckpt_name

    best_val_loss = float("inf")
    if resume_path:
        start_epoch, best_val_loss = _resume_from_checkpoint(
            resume_path=resume_path,
            encoder=encoder,
            decoder=decoder,
            critic=critic,
            gen_optimizer=gen_optimizer,
            critic_optimizer=critic_optimizer,
            gen_scheduler=gen_scheduler,
            critic_scheduler=critic_scheduler,
            scaler=scaler,
            device=device,
            logger=logger,
        )

    history = {
        "gen_loss": [],
        "gen_recon_loss": [],
        "gen_adv_loss": [],
        "critic_loss": [],
        "critic_wasserstein": [],
        "critic_gp": [],
        "val_gen_loss": [],
        "val_ssim": [],
        "val_psnr": [],
        "val_dice": [],
    }

    end_epoch = start_epoch + num_epochs - 1
    training_start_time = time.time()
    train_loader_len = len(train_loader)

    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch}/{end_epoch} started")

        encoder.train()
        decoder.train()
        critic.train()

        gen_loss_sum = 0.0
        gen_recon_sum = 0.0
        gen_adv_sum = 0.0
        critic_loss_sum = 0.0
        critic_wasserstein_sum = 0.0
        critic_gp_sum = 0.0

        gen_steps = 0
        critic_steps = 0
        last_critic_loss_value = float("nan")

        for batch_idx, batch in enumerate(train_loader):
            x = batch["input"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            x, y, m = tu.enforce_channels_last(x, y, m, device)

            _set_requires_grad(critic, True)
            _set_requires_grad(encoder, False)
            _set_requires_grad(decoder, False)

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    shared_fake = _run_generator(encoder, decoder, x)
            shared_fake = shared_fake.detach().to(dtype=y.dtype)

            for _ in range(n_critic):
                critic_optimizer.zero_grad(set_to_none=True)

                if recompute_fake_each_critic_step:
                    with torch.no_grad():
                        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                            fake = _run_generator(encoder, decoder, x)
                    fake = fake.detach().to(dtype=y.dtype)
                else:
                    fake = shared_fake

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    real_logits = critic(y)
                    fake_logits = critic(fake)
                    wasserstein = fake_logits.mean() - real_logits.mean()
                gp = _gradient_penalty(critic, y, fake)
                critic_loss = wasserstein + lambda_gp * gp

                if not torch.isfinite(critic_loss):
                    logger.warning("Non-finite critic loss encountered; skipping critic step")
                    continue

                critic_loss.backward()
                if critic_grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), critic_grad_clip)
                critic_optimizer.step()

                critic_loss_sum += float(critic_loss.item())
                critic_wasserstein_sum += float(wasserstein.item())
                critic_gp_sum += float(gp.item())
                critic_steps += 1
                last_critic_loss_value = float(critic_loss.item())

            _set_requires_grad(critic, False)
            _set_requires_grad(encoder, True)
            _set_requires_grad(decoder, True)

            gen_optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                preds = _run_generator(encoder, decoder, x)
                recon_loss, _, _, _ = loss_module(preds, y, m)
                adv_loss = -critic(preds).mean()
            gen_loss = recon_loss + lambda_adv * adv_loss

            if not torch.isfinite(gen_loss):
                logger.warning("Non-finite generator loss encountered; skipping generator step")
                continue

            if scaler.is_enabled():
                scaler.scale(gen_loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(gen_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        gen_params,
                        grad_clip,
                    )
                scaler.step(gen_optimizer)
                scaler.update()
            else:
                gen_loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        gen_params,
                        grad_clip,
                    )
                gen_optimizer.step()

            gen_loss_sum += float(gen_loss.item())
            gen_recon_sum += float(recon_loss.item())
            gen_adv_sum += float(adv_loss.item())
            gen_steps += 1

            if (batch_idx + 1) % max(1, train_loader_len // 10) == 0:
                progress = float(batch_idx + 1) / max(1, train_loader_len)
                progress_pct = progress * 100.0
                epoch_elapsed_s = time.time() - epoch_start
                est_epoch_total_s = epoch_elapsed_s / max(progress, 1e-8)
                est_epoch_remaining_s = max(0.0, est_epoch_total_s - epoch_elapsed_s)
                logger.info(
                    f"Epoch {epoch}/{end_epoch} - batch {batch_idx + 1}/{train_loader_len} | "
                    f"G: {gen_loss.item():.4f} | D: {last_critic_loss_value:.4f} | "
                    f"{progress_pct:.0f}% | Epoch Remaining: {_format_time(est_epoch_remaining_s)}"
                )

            _check_shutdown_requested(shutdown_flag_path, logger)

        train_gen_loss = gen_loss_sum / max(1, gen_steps)
        train_gen_recon = gen_recon_sum / max(1, gen_steps)
        train_gen_adv = gen_adv_sum / max(1, gen_steps)

        train_critic_loss = critic_loss_sum / max(1, critic_steps)
        train_critic_wasserstein = critic_wasserstein_sum / max(1, critic_steps)
        train_critic_gp = critic_gp_sum / max(1, critic_steps)

        encoder.eval()
        decoder.eval()
        critic.eval()

        val_gen_loss_sum = 0.0
        val_ssim_sum = 0.0
        val_psnr_sum = 0.0
        val_seg_loss_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                m = batch["mask"].to(device, non_blocking=True)
                x, y, m = tu.enforce_channels_last(x, y, m, device)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    preds = _run_generator(encoder, decoder, x)
                    recon_loss, _, _, seg_loss_val = loss_module(preds, y, m)
                    adv_loss = -critic(preds).mean()
                val_gen_loss = recon_loss + lambda_adv * adv_loss

                ssim_val, psnr_val = loss_module.compute_val_metrics(preds, y)

                val_gen_loss_sum += float(val_gen_loss.item())
                val_seg_loss_sum += float(seg_loss_val)
                val_ssim_sum += float(ssim_val)
                val_psnr_sum += float(psnr_val)
                val_steps += 1

                _check_shutdown_requested(shutdown_flag_path, logger)

        val_gen_loss_avg = val_gen_loss_sum / max(1, val_steps)
        val_ssim_avg = val_ssim_sum / max(1, val_steps)
        val_psnr_avg = val_psnr_sum / max(1, val_steps)
        val_seg_loss_avg = val_seg_loss_sum / max(1, val_steps)
        val_dice_avg = 1.0 - val_seg_loss_avg

        gen_scheduler.step(val_gen_loss_avg)
        critic_scheduler.step(train_critic_loss)

        if val_gen_loss_avg < best_val_loss:
            best_val_loss = val_gen_loss_avg
            _save_checkpoint(
                ckpt_path=best_ckpt_path,
                epoch=epoch,
                encoder=encoder,
                decoder=decoder,
                critic=critic,
                gen_optimizer=gen_optimizer,
                critic_optimizer=critic_optimizer,
                gen_scheduler=gen_scheduler,
                critic_scheduler=critic_scheduler,
                scaler=scaler,
                best_val_loss=best_val_loss,
            )
            logger.info(
                f"Saved best WGAN-GP checkpoint at epoch {epoch}: "
                f"val_gen_loss={best_val_loss:.6f}"
            )

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - training_start_time

        logger.info(
            f"Epoch {epoch:03d} | "
            f"G Loss: {train_gen_loss:.4f} | Recon: {train_gen_recon:.4f} | Adv: {train_gen_adv:.4f} | "
            f"D Loss: {train_critic_loss:.4f} | Wass: {train_critic_wasserstein:.4f} | GP: {train_critic_gp:.4f} | "
            f"Val G: {val_gen_loss_avg:.4f} | SSIM: {val_ssim_avg:.4f} | PSNR: {val_psnr_avg:.2f} | "
            f"Dice: {val_dice_avg:.4f}"
        )
        logger.info(
            f"LR G: {gen_optimizer.param_groups[0]['lr']:.2e} | "
            f"LR D: {critic_optimizer.param_groups[0]['lr']:.2e} | "
            f"Epoch: {_format_time(epoch_time)} | Elapsed: {_format_time(elapsed)}"
        )

        history["gen_loss"].append(train_gen_loss)
        history["gen_recon_loss"].append(train_gen_recon)
        history["gen_adv_loss"].append(train_gen_adv)
        history["critic_loss"].append(train_critic_loss)
        history["critic_wasserstein"].append(train_critic_wasserstein)
        history["critic_gp"].append(train_critic_gp)
        history["val_gen_loss"].append(val_gen_loss_avg)
        history["val_ssim"].append(val_ssim_avg)
        history["val_psnr"].append(val_psnr_avg)
        history["val_dice"].append(val_dice_avg)

        if cfg.get("wandb_mode", "online") != "disabled":
            wandb.log(
                {
                    "epoch": epoch,
                    "gen_loss": train_gen_loss,
                    "gen_recon_loss": train_gen_recon,
                    "gen_adv_loss": train_gen_adv,
                    "critic_loss": train_critic_loss,
                    "critic_wasserstein": train_critic_wasserstein,
                    "critic_gp": train_critic_gp,
                    "val_gen_loss": val_gen_loss_avg,
                    "SSIM": val_ssim_avg,
                    "PSNR": val_psnr_avg,
                    "Dice_Score": val_dice_avg,
                    "lr_gen": gen_optimizer.param_groups[0]["lr"],
                    "lr_disc": critic_optimizer.param_groups[0]["lr"],
                }
            )

        if tu.SHUTDOWN_REQUESTED:
            logger.info(f"Shutdown requested. Stopping at epoch {epoch}.")
            last_ckpt_path = ckpt_dir / f"epoch_{epoch}_wgan_gp_last.pth"
            _save_checkpoint(
                ckpt_path=last_ckpt_path,
                epoch=epoch,
                encoder=encoder,
                decoder=decoder,
                critic=critic,
                gen_optimizer=gen_optimizer,
                critic_optimizer=critic_optimizer,
                gen_scheduler=gen_scheduler,
                critic_scheduler=critic_scheduler,
                scaler=scaler,
                best_val_loss=best_val_loss,
            )
            logger.info(f"Saved last checkpoint to {last_ckpt_path}")
            break

    return encoder, decoder, critic, history
