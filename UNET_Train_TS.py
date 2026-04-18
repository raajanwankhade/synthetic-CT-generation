import os
import json
import argparse
secrets_path = 'secrets.json'
with open(secrets_path, 'r') as f:
    secrets = json.load(f)
os.environ["KAGGLE_USERNAME"] = secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = secrets["KAGGLE_KEY"]
os.environ["WANDB_API_KEY"] = secrets["wandb_API_KEY"]

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
from typing import Any, Dict, List, Optional, Tuple, cast

import wandb
import SimpleITK as sitk

from config import cfg as default_cfg
from UNET_model import build_resunet2d, build_resunet3d
import train_utils as tu
from dataset_class import MRCTDataset, build_dataloaders
from ts_dataclass import build_ts_dataloader as ts_dataloader
from UNET_Loss import LossModule

SHUTDOWN_FLAG_PATH = None


# ---------------------------------------------------------------------------
# Reproducibility + device helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42, cfg: dict | None = None):
    if cfg is None:
        cfg = default_cfg
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(cfg.get("deterministic", False))
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if not (0 <= local_rank < torch.cuda.device_count()):
            local_rank = 0
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Patch-start unpacking helper
# ---------------------------------------------------------------------------

def _unpack_patch_starts(raw) -> List[Tuple[int, int, int]]:
    """
    Convert DataLoader-collated patch_start into a list of (z, y, x) tuples.

    DataLoader default_collate turns a batch of Python tuples into a list of
    tensors — one tensor per tuple element. Handle that case plus the less
    common alternatives.
    """
    if (
        isinstance(raw, (list, tuple))
        and len(raw) == 3
        and torch.is_tensor(raw[0])
    ):
        # Most common: [tensor([z0,...]), tensor([y0,...]), tensor([x0,...])]
        zs = raw[0].tolist()
        ys = raw[1].tolist()
        xs = raw[2].tolist()
        return [(int(z), int(y), int(x)) for z, y, x in zip(zs, ys, xs)]
    if torch.is_tensor(raw):
        # (B, 3) tensor
        return [tuple(int(v) for v in row) for row in raw.tolist()]
    # Already a list/tuple of tuples
    return [tuple(int(v) for v in ps) for ps in raw]


# ---------------------------------------------------------------------------
# Checkpoint resume
# ---------------------------------------------------------------------------

def resume_from_ckpt(
    resume_path: str,
    encoder: nn.Module,
    decoder: nn.Module,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    use_amp: bool,
    num_epochs: int,
    start_epoch: int,
    logger=None,
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


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def perform_sanity_check(
    encoder: nn.Module,
    decoder: nn.Module,
    train_loader,
    device: torch.device,
    optimizer,
    scaler,
    loss_module: LossModule,
    amp_dtype: torch.dtype,
    use_amp: bool,
    grad_clip,
    sanity_steps: int,
    logger=None,
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
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()), grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()), grad_clip
                )
            optimizer.step()
    logger.info(f"✅ Sanity check passed ({sanity_steps} step(s))")


# ---------------------------------------------------------------------------
# Regular training loop
# ---------------------------------------------------------------------------

def regular_training(
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
    optimizer=None,
    scheduler=None,
    scaler=None,
    ckpt_mgr=None,
    return_state: bool = False,
    pseudo_label_cache=None,
    surrogate_head=None,
    surrogate_optimizer=None,
    surrogate_weight: float = 0.5,
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

    state_was_fresh = False
    if optimizer is None:
        optimizer = tu.build_optimizer(encoder, decoder, device, lr)
        state_was_fresh = True
    if scheduler is None:
        scheduler = tu.build_scheduler(optimizer, scheduler_patience)
        state_was_fresh = True
    if ckpt_mgr is None:
        ckpt_mgr = tu.CheckpointManager(
            run_dir / cfg["ckpt_dir"],
            monitor="val_loss", mode="min",
            filename=cfg["ckpt_filename"],
        )
        state_was_fresh = True
    if scaler is None:
        scaler_enabled = use_amp and amp_dtype == torch.float16
        grad_scaler_cls = getattr(getattr(torch, "amp", None), "GradScaler", torch.cuda.amp.GradScaler)
        scaler = grad_scaler_cls(enabled=scaler_enabled)
        state_was_fresh = True

    # Surrogate is only active when the cache has been populated (cycle 2+)
    surrogate_active = (
        pseudo_label_cache is not None
        and surrogate_head is not None
        and surrogate_optimizer is not None
        and pseudo_label_cache.is_populated
    )
    if surrogate_active:
        surrogate_head.to(device)
        surrogate_head.train()

    logger = logging.getLogger(__name__)
    logger.info(
        f"[surrogate] {'enabled' if surrogate_active else 'disabled'} | "
        f"cache_populated={int(pseudo_label_cache is not None and pseudo_label_cache.is_populated)}"
    )

    if resume_path is None and state_was_fresh:
        logger.info("🆕 Starting fresh training")
    elif resume_path is not None:
        start_epoch = resume_from_ckpt(
            resume_path, encoder, decoder, optimizer,
            scheduler, scaler, device, use_amp, num_epochs, start_epoch, logger=logger,
        )

    perform_sanity_check(
        encoder=encoder, decoder=decoder, train_loader=train_loader,
        device=device, optimizer=optimizer, scaler=scaler,
        loss_module=loss_module, amp_dtype=amp_dtype, use_amp=use_amp,
        grad_clip=grad_clip, sanity_steps=sanity_steps, logger=logger,
    )

    def _check_shutdown() -> bool:
        if not tu.SHUTDOWN_REQUESTED and shutdown_flag_path and os.path.exists(shutdown_flag_path):
            logger.warning("⏹️ Shutdown flag detected.")
            tu.SHUTDOWN_REQUESTED = True
        return tu.SHUTDOWN_REQUESTED

    def _fmt(seconds: float) -> str:
        h, r = divmod(int(seconds), 3600)
        m, s = divmod(r, 60)
        return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")

    history = {
        "train_loss": [], "val_loss": [], "val_seg_loss": [],
        "SSIM": [], "PSNR": [], "Dice_Score": [], "surrogate_loss": [],
    }
    end_epoch = start_epoch + num_epochs - 1
    t_start = time.time()
    epoch_times = []

    for epoch in range(start_epoch, end_epoch + 1):
        t_epoch = time.time()
        logger.info(f"Epoch {epoch}/{end_epoch} started")
        encoder.train()
        decoder.train()
        if surrogate_active:
            surrogate_head.train()

        train_loss_sum = 0.0
        surrogate_loss_sum = 0.0
        surrogate_attempted = 0
        surrogate_success = 0
        train_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            x = batch["input"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            x, y, m = tu.enforce_channels_last(x, y, m, device)

            optimizer.zero_grad(set_to_none=True)
            if surrogate_active:
                surrogate_optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                preds = decoder(encoder(x))
                total_loss, _, _, _ = loss_module(preds, y, m)
                surrogate_loss_val = 0.0

                if surrogate_active:
                    patient_names = list(batch["patient"]) if not isinstance(batch["patient"], list) else batch["patient"]
                    patch_starts = _unpack_patch_starts(batch["patch_start"])
                    patch_shape = tuple(preds.shape[2:])
                    surrogate_attempted += len(patient_names)

                    labels, valid_idx = [], []
                    for i, (pname, pstart) in enumerate(zip(patient_names, patch_starts)):
                        lbl = pseudo_label_cache.get_patch(pname, pstart, patch_shape)
                        if lbl is not None:
                            labels.append(lbl)
                            valid_idx.append(i)

                    if labels:
                        pseudo_batch = torch.stack(labels).to(device)
                        idx_t = torch.tensor(valid_idx, device=device)
                        seg_logits = surrogate_head(preds.index_select(0, idx_t))
                        s_loss = loss_module.surrogate_dice_loss(seg_logits, pseudo_batch)
                        total_loss = total_loss + surrogate_weight * s_loss
                        surrogate_loss_val = s_loss.item()
                        surrogate_success += len(valid_idx)

            if not torch.isfinite(total_loss):
                logger.warning("⚠️ Non-finite loss; skipping step")
                continue

            # All parameters (encoder, decoder, surrogate) in one backward pass
            all_params = list(encoder.parameters()) + list(decoder.parameters())
            if surrogate_active:
                all_params += list(surrogate_head.parameters())

            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    if surrogate_active:
                        scaler.unscale_(surrogate_optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
                scaler.step(optimizer)
                if surrogate_active and surrogate_loss_val > 0:
                    scaler.step(surrogate_optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
                optimizer.step()
                if surrogate_active and surrogate_loss_val > 0:
                    surrogate_optimizer.step()

            train_loss_sum += total_loss.item()
            surrogate_loss_sum += surrogate_loss_val
            train_steps += 1
            _check_shutdown()

            if (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
                logger.info(
                    f"Epoch {epoch}/{end_epoch} - batch {batch_idx+1}/{len(train_loader)} "
                    f"- loss={total_loss.item():.4f}"
                )

        train_loss = train_loss_sum / max(1, train_steps)
        surrogate_loss_avg = surrogate_loss_sum / max(1, train_steps)

        # --- Validation ---
        encoder.eval()
        decoder.eval()
        val_loss_sum, ssim_sum, psnr_sum, seg_loss_sum = 0.0, 0.0, 0.0, 0.0
        val_steps = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                x = batch["input"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                m = batch["mask"].to(device, non_blocking=True)
                x, y, m = tu.enforce_channels_last(x, y, m, device)
                preds = decoder(encoder(x))
                total_loss, _, _, seg_loss_val = loss_module(preds, y, m)
                val_loss_sum += total_loss.item()
                seg_loss_sum += seg_loss_val
                s, p = loss_module.compute_val_metrics(preds, y)
                ssim_sum += s
                psnr_sum += p
                val_steps += 1
                if (batch_idx + 1) % max(1, len(val_loader) // 10) == 0:
                    logger.info(
                        f"Epoch {epoch}/{end_epoch} - val batch {batch_idx+1}/{len(val_loader)} "
                        f"- loss={total_loss.item():.4f}"
                    )
                _check_shutdown()

        val_loss = val_loss_sum / max(1, val_steps)
        seg_loss_avg = seg_loss_sum / max(1, val_steps)
        dice_score_avg = 1.0 - seg_loss_avg
        ssim_avg = ssim_sum / max(1, val_steps)
        psnr_avg = psnr_sum / max(1, val_steps)

        val_metrics = {"val_loss": val_loss, "SSIM": ssim_avg, "PSNR": psnr_avg, "Dice_Score": dice_score_avg}
        scheduler.step(val_loss)
        ckpt_mgr.update(epoch, encoder, decoder, optimizer, scheduler, scaler, val_metrics)

        epoch_dur = time.time() - t_epoch
        epoch_times.append(epoch_dur)
        elapsed = time.time() - t_start
        remaining = (sum(epoch_times) / len(epoch_times)) * (end_epoch - epoch)

        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train={train_loss:.4f} | Val={val_loss:.4f} | "
            f"SSIM={ssim_avg:.4f} | PSNR={psnr_avg:.2f} | Dice={dice_score_avg:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.1e}"
        )
        if surrogate_active:
            logger.info(
                f"[surrogate] coverage={surrogate_success}/{surrogate_attempted} "
                f"({100*surrogate_success/max(1,surrogate_attempted):.1f}%) | "
                f"loss={surrogate_loss_avg:.4f}"
            )
        logger.info(f"Time - epoch={_fmt(epoch_dur)} | elapsed={_fmt(elapsed)} | remaining={_fmt(remaining)} (est.)")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_seg_loss"].append(seg_loss_avg)
        history["SSIM"].append(ssim_avg)
        history["PSNR"].append(psnr_avg)
        history["Dice_Score"].append(dice_score_avg)
        history["surrogate_loss"].append(surrogate_loss_avg)

        if wandb_log:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_seg_loss": seg_loss_avg,
                "SSIM": ssim_avg,
                "PSNR": psnr_avg,
                "Dice_Score": dice_score_avg,
                "surrogate_loss": surrogate_loss_avg,
                "surrogate_active": float(surrogate_active),
                "surrogate_coverage": float(surrogate_success) / max(1, surrogate_attempted),
                "lr": optimizer.param_groups[0]["lr"],
            })

            log_every = int(cfg.get("log_images_every", 0))
            save_every = int(cfg.get("save_vis_every", 0))
            should_log = log_every > 0 and (epoch % log_every == 0)
            should_save = save_every > 0 and (epoch % save_every == 0)
            if should_log or should_save:
                fig = tu.visualize_batch(encoder, decoder, val_loader, device=device, num_samples=4)
                if should_log:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    wandb.log({f"epoch_{epoch}_output": wandb.Image(Image.open(buf), caption=f"Epoch {epoch}")})
                if should_save:
                    fig.savefig(str(run_dir / f"epoch_{epoch}_{cfg['Visualisation_Fig']}"))
                plt.close(fig)

        if tu.SHUTDOWN_REQUESTED:
            logger.info(f"[shutdown] Stopping at epoch {epoch}.")
            ckpt_mgr.save_epoch(run_dir / f"epoch_{epoch}.pt", epoch, encoder, decoder, optimizer, scheduler, scaler)
            break

    if return_state:
        return encoder, decoder, history, optimizer, scheduler, scaler, ckpt_mgr
    return encoder, decoder, history


# ---------------------------------------------------------------------------
# Helpers for TS phase
# ---------------------------------------------------------------------------

def _binary_dice(pred_seg: np.ndarray, gt_seg: np.ndarray, eps: float = 1e-6) -> float:
    pred_fg = pred_seg > 0
    gt_fg = gt_seg > 0
    intersection = float(np.logical_and(pred_fg, gt_fg).sum())
    denom = float(pred_fg.sum() + gt_fg.sum())
    return (2.0 * intersection + eps) / (denom + eps)


def _to_numpy_segmentation(seg_obj) -> np.ndarray:
    if isinstance(seg_obj, tuple):
        seg_obj = seg_obj[0]
    if isinstance(seg_obj, np.ndarray):
        return seg_obj
    if isinstance(seg_obj, sitk.Image):
        return sitk.GetArrayFromImage(seg_obj)
    if isinstance(seg_obj, (str, Path)):
        p = Path(seg_obj)
        if p.exists():
            return sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
    if hasattr(seg_obj, "get_fdata"):
        return np.asarray(seg_obj.get_fdata())
    return np.asarray(seg_obj)


def _sitk_grids_differ(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() != b.GetSize()
        or a.GetSpacing() != b.GetSpacing()
        or a.GetOrigin() != b.GetOrigin()
        or a.GetDirection() != b.GetDirection()
    )


# ---------------------------------------------------------------------------
# TS phase
# ---------------------------------------------------------------------------

def _run_ts_phase(
    encoder: nn.Module,
    decoder: nn.Module,
    ts_loader,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    run_dir: Path,
    cfg: dict,
    pseudo_label_cache=None,
    logger=None,
) -> Dict[str, float]:
    if logger is None:
        logger = logging.getLogger(__name__)

    from totalsegmentator.python_api import totalsegmentator

    ds = ts_loader.dataset
    required = [
        "patient_to_dataset_indices",
        "stitch_patient_predictions",
        "get_patient_reference_image",
        "get_patient_modality_reference_image",
        "get_patient_segmentation",
    ]
    if not all(hasattr(ds, attr) for attr in required):
        raise TypeError(f"TS dataset missing required methods: {required}")

    # --- Pass 1: predict all patches per patient (no gradient needed) ---
    pred_patches_by_patient: Dict[str, List[Optional[np.ndarray]]] = {}
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for batch in ts_loader:
            x = batch["input"].to(device, non_blocking=True)
            y_dummy = batch["label"].to(device, non_blocking=True)
            m_dummy = torch.ones_like(y_dummy)
            x, _, _ = tu.enforce_channels_last(x, y_dummy, m_dummy, device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                preds = decoder(encoder(x))

            patient_names = batch["patient"]
            if isinstance(patient_names, str):
                patient_names = [patient_names]
            patch_indices = batch["patch_idx"]
            if torch.is_tensor(patch_indices):
                patch_indices = patch_indices.tolist()

            pred_np = preds.detach().float().cpu().numpy()
            for i, patient_name in enumerate(patient_names):
                patch_idx = int(patch_indices[i])
                n_patches = len(ds.patient_to_dataset_indices[patient_name])
                if patient_name not in pred_patches_by_patient:
                    pred_patches_by_patient[patient_name] = [None] * n_patches
                patch = pred_np[i]
                if patch.ndim == 4 and patch.shape[0] == 1:
                    patch = patch[0]
                pred_patches_by_patient[patient_name][patch_idx] = (
                    np.ascontiguousarray(patch, dtype=np.float32)
                )

    # --- Pass 2: stitch, run TS, update cache ---
    ts_tmp_dir = cfg.get("ts_tmp_dir", None)
    use_ramdisk = bool(cfg.get("ts_use_ramdisk", True))
    if ts_tmp_dir:
        ts_input_dir = Path(str(ts_tmp_dir))
    elif use_ramdisk and Path("/dev/shm").exists():
        ts_input_dir = Path("/dev/shm") / f"ts_inputs_{os.getpid()}"
    else:
        ts_input_dir = run_dir / "ts_inputs"
    ts_input_dir.mkdir(parents=True, exist_ok=True)

    # Only update the cache if TS Dice is above this floor.
    # When the model is early in training, TS sees noise and produces
    # meaningless pseudo-labels. Gate on a minimum quality.
    min_dice_for_cache = float(cfg.get("ts_min_dice_for_cache", 0.05))

    ts_dice_values: List[float] = []

    for patient_name, patch_slots in pred_patches_by_patient.items():
        if any(p is None for p in patch_slots):
            logger.warning(f"[TS] {patient_name}: missing patches, skipping")
            continue

        # Stitch into full volume — this lives in MRI voxel space because the
        # TS dataset extracts patches from the MRI and uses MRI shape for stitching.
        pred_patches = [cast(np.ndarray, p) for p in patch_slots]
        stitched_sct = ds.stitch_patient_predictions(patient_name, pred_patches)

        # Tag the stitched volume with MRI spatial metadata (correct for MRI space).
        # Previously this used the CT reference, which is wrong when MRI and CT
        # have different voxel grids.
        mri_ref = ds.get_patient_modality_reference_image(patient_name)
        input_image = sitk.GetImageFromArray(stitched_sct.astype(np.float32))
        input_image.CopyInformation(mri_ref)

        input_path = ts_input_dir / f"{patient_name}_sct.nii.gz"
        sitk.WriteImage(input_image, str(input_path))

        output_img = totalsegmentator(str(input_path), ml=True, task="total", fast=True, quiet=True)
        pred_seg = _to_numpy_segmentation(output_img)  # shape = MRI voxel space

        if not cfg.get("keep_ts_inputs", False):
            input_path.unlink(missing_ok=True)

        # Resample pseudo-label from MRI voxel space into training CT space.
        # If MRI and CT share the same voxel grid (same shape, spacing, origin),
        # the resample is a no-op and pred_seg is used directly.
        ct_ref = ds.get_patient_reference_image(
            patient_name,
            use_preproc_override=bool(cfg.get("use_preproc_files", False)),
        )
        if _sitk_grids_differ(mri_ref, ct_ref):
            pred_seg_img = sitk.GetImageFromArray(pred_seg.astype(np.uint8, copy=False))
            pred_seg_img.CopyInformation(mri_ref)
            pred_seg_ct = sitk.GetArrayFromImage(
                sitk.Resample(
                    pred_seg_img, ct_ref,
                    sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
                )
            )
        else:
            pred_seg_ct = pred_seg  # already in CT space

        # Monitoring Dice vs ground-truth (gt is in CT space)
        gt_seg = ds.get_patient_segmentation(patient_name)
        if gt_seg is None:
            logger.warning(f"[TS] {patient_name}: no ground-truth segmentation, skipping Dice")
            if pseudo_label_cache is not None:
                pseudo_label_cache.update(patient_name, pred_seg_ct)
            continue

        if pred_seg_ct.shape != gt_seg.shape:
            logger.warning(
                f"[TS] {patient_name}: shape mismatch after resample "
                f"pred={pred_seg_ct.shape} gt={gt_seg.shape}, skipping"
            )
            continue

        dice_score = _binary_dice(pred_seg_ct, gt_seg)
        ts_dice_values.append(float(dice_score))

        cache_updated = False
        if pseudo_label_cache is not None and dice_score >= min_dice_for_cache:
            pseudo_label_cache.update(patient_name, pred_seg_ct)
            cache_updated = True

        logger.info(
            f"[TS] {patient_name} | dice={dice_score:.4f} | "
            f"cache={'updated' if cache_updated else f'skipped (< {min_dice_for_cache:.2f})'}"
        )

    ts_dice_avg = float(np.mean(ts_dice_values)) if ts_dice_values else 0.0
    cache_size = len(pseudo_label_cache._volumes) if pseudo_label_cache is not None else 0
    return {
        "ts_dice": ts_dice_avg,
        "ts_patients": float(len(ts_dice_values)),
        "ts_cache_size": float(cache_size),
    }


# ---------------------------------------------------------------------------
# History plotting
# ---------------------------------------------------------------------------

def _plot_history(history: Dict[str, List[float]], run_dir: Path, cfg: dict, logger=None) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)

    metric_specs = [
        ("train_loss", "val_loss", "Train vs Validation Loss", "Loss", "epoch"),
        ("val_seg_loss", None, "Validation Segmentation Loss", "Seg Loss", "epoch"),
        ("SSIM", None, "SSIM", "SSIM", "epoch"),
        ("PSNR", None, "PSNR", "PSNR (dB)", "epoch"),
        ("Dice_Score", None, "Validation Dice Score", "Dice", "epoch"),
        ("surrogate_loss", None, "Surrogate Loss", "Surrogate Loss", "epoch"),
        ("ts_dice", None, "TS Dice (monitoring)", "TS Dice", "ts"),
    ]

    active_specs = [
        spec for spec in metric_specs
        if bool(history.get(spec[0])) or bool(spec[1] and history.get(spec[1]))
    ]
    if not active_specs:
        logger.warning("No history values available for plotting.")
        return

    fig, axes = plt.subplots(len(active_specs), 1, figsize=(14, 3.8 * len(active_specs)))
    if len(active_specs) == 1:
        axes = [axes]

    for ax, (key_a, key_b, title, ylabel, x_kind) in zip(axes, active_specs):
        x_label = "TS Cycle" if x_kind == "ts" else "Epoch"
        if key_a and history.get(key_a):
            ax.plot(range(1, len(history[key_a]) + 1), history[key_a], label=key_a, linewidth=2)
        if key_b and history.get(key_b):
            ax.plot(range(1, len(history[key_b]) + 1), history[key_b], label=key_b, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    out_path = run_dir / cfg["Loss_Plot_name"]
    fig.savefig(str(out_path))
    plt.close(fig)
    logger.info(f"Saved training curves: {out_path}")


# ---------------------------------------------------------------------------
# train_model: alternating regular + TS cycles
# ---------------------------------------------------------------------------

def train_model(
    encoder: nn.Module,
    decoder: nn.Module,
    train_loader,
    val_loader,
    ts_loader,
    TS_epochs: int,
    regular_epochs: int,
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
    train_patient_names: Optional[List[str]] = None,
):
    if cfg is None:
        raise ValueError("cfg must be provided")

    from ts_pseudo_label_cache import TSPseudoLabelCache
    from surrogate_seg_head import SurrogateSegHead

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if loss_module is None:
        loss_module = LossModule(device)

    TS_epochs = max(1, int(TS_epochs))
    regular_epochs = max(1, int(regular_epochs))
    use_amp = device.type == "cuda"
    amp_dtype = tu.get_amp_dtype(cfg)
    logger = logging.getLogger(__name__)

    surrogate_weight = float(cfg.get("surrogate_weight", 0.5))
    ts_seg_classes = cfg.get("ts_seg_classes", "binary")

    if isinstance(ts_seg_classes, str) and ts_seg_classes.strip().lower() == "binary":
        use_binary, selected_classes, n_classes = True, None, 1
    elif isinstance(ts_seg_classes, str) and ts_seg_classes.strip().lower() == "all":
        use_binary, selected_classes, n_classes = False, None, 104
    elif isinstance(ts_seg_classes, (list, tuple)):
        selected_classes = [int(v) for v in ts_seg_classes]
        use_binary, n_classes = False, len(selected_classes)
    else:
        try:
            selected_classes = [int(v) for v in json.loads(ts_seg_classes)]
            use_binary, n_classes = False, len(selected_classes)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                "ts_seg_classes must be 'binary', 'all', or a JSON list of class indices"
            ) from exc

    logger.info(
        f"[surrogate] class_mode={'binary' if use_binary else 'multiclass'} | "
        f"n_classes={n_classes} | "
        f"selected={'all' if not use_binary and selected_classes is None else selected_classes}"
    )

    pseudo_label_cache = TSPseudoLabelCache(
        cache_dir=run_dir / "ts_cache" if cfg.get("persist_ts_cache", True) else None,
        use_binary=use_binary,
        selected_classes=selected_classes,
    )
    surrogate_head = SurrogateSegHead(n_classes=n_classes)
    surrogate_optimizer = torch.optim.Adam(
        surrogate_head.parameters(),
        lr=float(cfg.get("surrogate_lr", 1e-3)),
    )

    history = {
        "train_loss": [], "val_loss": [], "val_seg_loss": [],
        "SSIM": [], "PSNR": [], "Dice_Score": [],
        "surrogate_loss": [], "ts_dice": [],
    }

    regular_optimizer = None
    regular_scheduler = None
    regular_scaler = None
    regular_ckpt_mgr = None
    rolling_start_epoch = int(start_epoch)
    ts_seed_base = int(cfg.get("seed", 42))

    for ts_cycle in range(1, TS_epochs + 1):
        logger.info(f"[TS cycle] {ts_cycle}/{TS_epochs} | regular_epochs={regular_epochs}")

        (
            encoder, decoder, reg_history,
            regular_optimizer, regular_scheduler, regular_scaler, regular_ckpt_mgr,
        ) = regular_training(
            encoder=encoder, decoder=decoder,
            train_loader=train_loader, val_loader=val_loader,
            num_epochs=regular_epochs, start_epoch=rolling_start_epoch,
            lr=lr, scheduler_patience=scheduler_patience,
            device=device, wandb_log=wandb_log, loss_module=loss_module,
            resume_path=resume_path if ts_cycle == 1 else None,
            grad_clip=grad_clip, sanity_steps=sanity_steps if ts_cycle == 1 else 0,
            cfg=cfg, run_dir=run_dir, shutdown_flag_path=shutdown_flag_path,
            optimizer=regular_optimizer, scheduler=regular_scheduler,
            scaler=regular_scaler, ckpt_mgr=regular_ckpt_mgr, return_state=True,
            pseudo_label_cache=pseudo_label_cache,
            surrogate_head=surrogate_head,
            surrogate_optimizer=surrogate_optimizer,
            surrogate_weight=surrogate_weight,
        )

        for key in ("train_loss", "val_loss", "val_seg_loss", "SSIM", "PSNR", "Dice_Score", "surrogate_loss"):
            history[key].extend(reg_history.get(key, []))

        rolling_start_epoch += regular_epochs
        if tu.SHUTDOWN_REQUESTED:
            break

        # Build (possibly randomised) TS loader for this cycle
        active_ts_loader = ts_loader
        if ts_loader is not None and bool(cfg.get("ts_randomize_each_cycle", True)):
            ts_patch_size = tuple(cfg.get("ts_patch_size", cfg["patch_size"]))
            ts_stride = tuple(cfg.get("ts_stride", ts_patch_size))
            ts_seed = ts_seed_base + ts_cycle
            active_ts_loader = ts_dataloader(
                dataset_root=cfg["root"],
                fraction=float(cfg.get("ts_fraction", 0.2)),
                modality=cfg["modality"],
                patch_size=ts_patch_size,
                stride=ts_stride,
                use_preproc_files=bool(cfg.get("ts_use_preproc_files", False)),
                batch_size=int(cfg.get("ts_batch_size", 1)),
                num_workers=int(cfg.get("ts_num_workers", 0)),
                pin_memory=(device.type == "cuda"),
                shuffle=False,
                seed=ts_seed,
                cache_size=int(cfg.get("ts_cache_size", cfg.get("cache_size", 2))),
                include_patient_names=train_patient_names,
            )
            logger.info(
                f"[TS cycle] randomised loader | seed={ts_seed} | "
                f"patients={len(active_ts_loader.dataset.patients)} | "
                f"patches={len(active_ts_loader.dataset)}"
            )

        if active_ts_loader is not None:
            ts_metrics = _run_ts_phase(
                encoder=encoder, decoder=decoder,
                ts_loader=active_ts_loader,
                device=device, amp_dtype=amp_dtype, use_amp=use_amp,
                run_dir=run_dir, cfg=cfg,
                pseudo_label_cache=pseudo_label_cache,
                logger=logger,
            )
            history["ts_dice"].append(ts_metrics["ts_dice"])

            if wandb_log:
                wandb.log({
                    "ts_cycle": ts_cycle,
                    "ts_dice": ts_metrics["ts_dice"],
                    "ts_patients": ts_metrics["ts_patients"],
                    "ts_cache_size": ts_metrics["ts_cache_size"],
                }, step=len(history["train_loss"]))

            logger.info(
                f"[TS cycle] {ts_cycle}/{TS_epochs} done | "
                f"dice={ts_metrics['ts_dice']:.4f} | "
                f"patients={int(ts_metrics['ts_patients'])} | "
                f"cache_size={int(ts_metrics['ts_cache_size'])}"
            )
        else:
            logger.warning("[TS cycle] No TS dataloader; skipping TS phase")

        if tu.SHUTDOWN_REQUESTED:
            break

    return encoder, decoder, history


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(cfg: dict):
    global SHUTDOWN_FLAG_PATH
    SHUTDOWN_FLAG_PATH = os.path.join(cfg["run_dir"], "stop.flag")

    run_dir = Path(cfg["run_dir"])
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Starting UNET training")
    logger.info("=" * 60)

    SEED = 42
    seed_everything(SEED, cfg)

    device = _select_device()
    use_amp = device.type == "cuda"
    logger.info(f"Device: {device} | AMP: {use_amp} | CUDA devices: {torch.cuda.device_count()}")

    if device.type == "cuda" and cfg.get("allow_tf32", True):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        logger.warning("⚠️ torch.set_float32_matmul_precision not available")

    root = cfg["root"]
    MAX_EPOCHS = cfg["epochs"]
    IS_3D = str(cfg.get("num_adjacent_slices", 0)).lower() == "3d"

    tu.init_wandb(cfg, SEED, use_amp, IS_3D)

    # Dataset
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
        cache_size=cfg["cache_size"],
    )
    logger.info(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    logger.info(
        f"Sample shapes - input: {sample['input'].shape}, "
        f"label: {sample['label'].shape}, mask: {sample['mask'].shape}"
    )

    train_loader, val_loader, train_ds, val_ds, train_patients, val_patients = build_dataloaders(
        dataset, cfg, SEED, device
    )
    train_patient_names = sorted({Path(p).name for p in train_patients})
    logger.info(f"Patients: {len(train_patients)} train | {len(val_patients)} val")
    logger.info(f"Samples: {len(train_ds)} train | {len(val_ds)} val")

    # TS dataloader (3D only)
    ts_loader = None
    if IS_3D:
        ts_patch_size = tuple(cfg.get("ts_patch_size", cfg["patch_size"]))
        ts_stride = tuple(cfg.get("ts_stride", ts_patch_size))
        ts_loader = ts_dataloader(
            dataset_root=root,
            fraction=float(cfg.get("ts_fraction", 0.2)),
            modality=cfg["modality"],
            patch_size=ts_patch_size,
            stride=ts_stride,
            use_preproc_files=bool(cfg.get("ts_use_preproc_files", False)),
            batch_size=int(cfg.get("ts_batch_size", 1)),
            num_workers=int(cfg.get("ts_num_workers", 0)),
            pin_memory=(device.type == "cuda"),
            shuffle=False,
            seed=SEED,
            cache_size=int(cfg.get("ts_cache_size", cfg.get("cache_size", 2))),
            include_patient_names=train_patient_names,
        )
        logger.info(
            f"TS dataloader | fraction={cfg.get('ts_fraction', 0.2):.3f} | "
            f"patients={len(ts_loader.dataset.patients)} | "
            f"patches={len(ts_loader.dataset)}"
        )
    else:
        logger.warning("2D mode: TS dataloader skipped (requires 3D patches)")

    # Model
    if IS_3D:
        enc, dec = build_resunet3d(cfg["model3d_cfg"])
        logger.info(f"3D model config: {cfg['model3d_cfg']}")
    else:
        enc, dec = build_resunet2d(cfg["model2d_cfg"])
        logger.info(f"2D model config: {cfg['model2d_cfg']}")

    enc, dec, compile_ok = tu.maybe_compile_models(enc, dec, cfg, device)
    logger.info("✅ Models compiled" if compile_ok else "⚠️ Models not compiled; eager mode")

    # Loss + training
    loss_module = LossModule(
        device,
        ssim_down=int(cfg.get("ssim_downsample", 1)),
        ssim_weight=cfg["lambda_ssim"],
        seg_weight=cfg.get("lambda_seg", 0.1),
    )
    trained_enc, trained_dec, history = train_model(
        encoder=enc, decoder=dec,
        train_loader=train_loader, val_loader=val_loader, ts_loader=ts_loader,
        TS_epochs=int(cfg.get("TS_epochs", 1)),
        regular_epochs=int(cfg.get("regular_epochs", MAX_EPOCHS)),
        start_epoch=cfg["start"], lr=cfg["lr"],
        device=device,
        wandb_log=(cfg["wandb_mode"] != "disabled"),
        loss_module=loss_module,
        scheduler_patience=cfg["scheduler_patience"],
        grad_clip=cfg["grad_clip"],
        sanity_steps=cfg["sanity_steps"],
        resume_path=cfg["resume_path"],
        cfg=cfg, run_dir=run_dir,
        shutdown_flag_path=SHUTDOWN_FLAG_PATH,
        train_patient_names=train_patient_names,
    )

    _plot_history(history, run_dir, cfg, logger=logger)
    wandb.finish()
    tu.visualize_batch(trained_enc, trained_dec, val_loader)

    if tu.SHUTDOWN_REQUESTED:
        logger.info("[shutdown] Removing flag file.")
        os.remove(SHUTDOWN_FLAG_PATH)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _infer_type(value):
    if value is None:
        return str
    if isinstance(value, bool):
        return str
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    return str


def _parse_value(value_str: str, default_value):
    if default_value is None:
        if value_str.lower() == "none":
            return None
        if value_str.lower() in ("true", "false"):
            return value_str.lower() == "true"
        for t in (int, float):
            try:
                return t(value_str)
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
        try:
            result = json.loads(value_str)
        except json.JSONDecodeError:
            result = []
            for item in value_str.split(","):
                item = item.strip()
                for t in (int, float):
                    try:
                        result.append(t(item))
                        break
                    except ValueError:
                        pass
                else:
                    result.append(item)
        return tuple(result) if isinstance(default_value, tuple) else result
    if isinstance(default_value, dict):
        return json.loads(value_str)
    if value_str.lower() in ("true", "false"):
        return value_str.lower() == "true"
    for t in (int, float):
        try:
            return t(value_str)
        except ValueError:
            pass
    return value_str


def build_argparser(cfg: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UNET Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for key, value in cfg.items():
        if key == "run_dir":
            parser.add_argument(f"--{key}", type=str, required=True,
                                help=f"(REQUIRED) Run directory")
        elif isinstance(value, bool):
            parser.add_argument(f"--{key}", type=str, default=str(value),
                                choices=["true", "false", "True", "False", "1", "0"])
        elif isinstance(value, (list, tuple, dict)):
            parser.add_argument(f"--{key}", type=str, default=json.dumps(value),
                                help=f"JSON string. Default: {json.dumps(value)}")
        else:
            arg_type = _infer_type(value)
            parser.add_argument(f"--{key}", type=arg_type if value is not None else str,
                                default=value if value is not None else "None",
                                help=f"Default: {value}")
    parser.add_argument("--log_file", type=str, default="train.log")
    if "TS_epochs" not in cfg:
        parser.add_argument("--TS_epochs", type=int, default=1)
    if "regular_epochs" not in cfg:
        parser.add_argument("--regular_epochs", type=int, default=int(cfg.get("epochs", 1)))
    return parser


def update_cfg_from_args(cfg: dict, args: argparse.Namespace) -> dict:
    updated = cfg.copy()
    for key, default_value in cfg.items():
        if hasattr(args, key):
            v = getattr(args, key)
            updated[key] = _parse_value(v, default_value) if isinstance(v, str) else v
    for key in ("TS_epochs", "regular_epochs"):
        if hasattr(args, key):
            updated[key] = int(getattr(args, key))
        elif key in cfg:
            updated[key] = int(cfg[key])
    if "regular_epochs" not in updated and "epochs" in cfg:
        updated["regular_epochs"] = int(cfg["epochs"])
    return updated


def save_config_to_run_dir(cfg: dict, run_dir: Path, logger=None) -> Path:
    if logger is None:
        logger = logging.getLogger(__name__)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.py"
    lines = ["cfg = {\n"]
    for key, value in cfg.items():
        lines.append(f'    "{key}": {repr(value)},\n')
    lines.append("}\n")
    config_path.write_text("".join(lines))
    logger.info(f"✅ Config saved to: {config_path}")
    return config_path


if __name__ == "__main__":
    parser = build_argparser(default_cfg)
    args = parser.parse_args()
    cfg = update_cfg_from_args(default_cfg, args)

    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / args.log_file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w")],
    )
    logger = logging.getLogger(__name__)

    config_path = save_config_to_run_dir(cfg, run_dir, logger=logger)
    os.environ["CONFIG_PATH"] = str(config_path)
    logger.info(f"📁 CONFIG_PATH set to: {config_path}")

    SHUTDOWN_FLAG_PATH = os.path.join(cfg["run_dir"], "stop.flag")
    main(cfg)