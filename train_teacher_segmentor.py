import argparse
import importlib.util
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from monai.losses.dice import DiceLoss
from monai.networks.nets.unet import UNet

import train_utils as tu
from dataset_class import MRCTDataset, build_dataloaders


DEFAULT_CONFIG_PATH = "config.py"
DEFAULT_SEED = 42
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-b1ebaa2d-6256-57d7-a497-6c7fc5dd254b"

def _load_cfg(config_path: str | None = None):
    path = config_path or os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)
    spec = importlib.util.spec_from_file_location("config", path)
    print(f"[train_teacher_segmentor.py]Loading config from {path} ...")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cfg


cfg = _load_cfg()


def _parse_args():
    parser = argparse.ArgumentParser(description="Train teacher segmentation model with config overrides")
    parser.add_argument("--run_dir", type=str, help="Override run directory")
    parser.add_argument("--patch_size", type=str, help="Override patch size (e.g., '32,192,192')")
    parser.add_argument("--num_adjacent_slices", type=str, help="Override num_adjacent_slices (e.g., '3d' or '3')")
    parser.add_argument("--use_pruned_data", type=lambda x: x.lower() in ['true', '1', 'yes'], help="Override use_pruned_data (true/false)")
    parser.add_argument("--use_preproc_files", type=lambda x: x.lower() in ['true', '1', 'yes'], help="Override use_preproc_files (true/false)")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--log_file", type=str, default="teacher_train.log", help="Log file name (default: teacher_train.log)")
    return parser.parse_args()


def _override_cfg_with_args(cfg: dict, args):
    """Override config with command-line arguments if provided."""
    if args.run_dir is not None:
        cfg["run_dir"] = args.run_dir
    if args.patch_size is not None:
        cfg["patch_size"] = tuple(int(x) for x in args.patch_size.split(','))
    if args.num_adjacent_slices is not None:
        # Handle both numeric and "3d" values
        if args.num_adjacent_slices.lower() == "3d":
            cfg["num_adjacent_slices"] = "3d"
        else:
            cfg["num_adjacent_slices"] = int(args.num_adjacent_slices)
    if args.use_pruned_data is not None:
        cfg["use_pruned_data"] = args.use_pruned_data
    if args.use_preproc_files is not None:
        cfg["use_preproc_files"] = args.use_preproc_files
    if args.epochs is not None:
        cfg["epochs"] = args.epochs


def _setup_logging(run_dir: Path, log_filename: str = "teacher_train.log"):
    """Set up logging to both console and file."""
    log_file = run_dir / log_filename
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def seed_everything(seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(cfg.get("deterministic", False))
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))


def _select_device():
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if not (0 <= local_rank < torch.cuda.device_count()):
            local_rank = 0
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _teacher_transform(sample):
    ct = sample["label"]
    mask = (sample["mask"] > 0.5).float()
    return {
        "image": ct,
        "label": mask,
        "patient": sample.get("patient"),
        "region": sample.get("region"),
        "slice_idx": sample.get("slice_idx"),
    }


def _maybe_channels_last(x: torch.Tensor, y: torch.Tensor, is_3d: bool, cfg: dict, device: torch.device):
    if not cfg.get("channels_last", False) or device.type != "cuda":
        return x, y
    if is_3d and hasattr(torch, "channels_last_3d"):
        return (
            x.contiguous(memory_format=torch.channels_last_3d),
            y.contiguous(memory_format=torch.channels_last_3d),
        )
    return (
        x.contiguous(memory_format=torch.channels_last),
        y.contiguous(memory_format=torch.channels_last),
    )


def _build_teacher_unet(dataset: MRCTDataset, cfg: dict, device: torch.device) -> nn.Module:
    spatial_dims = 3 if dataset.is_3d else 2
    in_channels = 1 if dataset.is_3d else dataset.num_adjacent

    teacher_cfg = cfg.get("teacher_cfg", {})
    stage_channels = tuple(teacher_cfg.get("stage_channels", [32, 64, 128, 256]))
    strides = tuple([2] * (len(stage_channels) - 1))

    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=int(teacher_cfg.get("out_ch", 1)),
        channels=stage_channels,
        strides=strides,
        num_res_units=int(teacher_cfg.get("num_res_units", 2)),
        norm=teacher_cfg.get("norm", "instance"),
    ).to(device)

    if cfg.get("channels_last", False):
        if dataset.is_3d and hasattr(torch, "channels_last_3d"):
            model = model.to(memory_format=torch.channels_last_3d)
        elif device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)

    if cfg.get("torch_compile", False) and hasattr(torch, "compile"):
        compile_mode = cfg.get("torch_compile_mode", "reduce-overhead")
        compile_backend = cfg.get("torch_compile_backend", "inductor")
        try:
            model = torch.compile(model, mode=compile_mode, backend=compile_backend)
            logging.getLogger(__name__).info(f"Teacher model compiled (mode={compile_mode}, backend={compile_backend})")
        except Exception as e:
            logging.getLogger(__name__).warning(f"torch.compile disabled for teacher model: {e}")

    return model


def _build_optimizer(model: nn.Module, device: torch.device, lr: float):
    try:
        return torch.optim.AdamW(model.parameters(), lr=lr, fused=(device.type == "cuda"))
    except TypeError:
        return torch.optim.AdamW(model.parameters(), lr=lr)


def _load_resume_if_available(
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    resume_path: str | None,
    device: torch.device,
):
    if not resume_path:
        return 1, float("inf")
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"resume_path does not exist: {resume_path}")

    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "scaler" in ckpt and scaler.is_enabled():
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    logging.getLogger(__name__).info(f"Resumed teacher training from epoch {start_epoch}")
    return start_epoch, best_val_loss


def _save_teacher_checkpoint(path: Path, epoch: int, model: nn.Module, optimizer, scheduler, scaler, best_val_loss: float):
    torch.save(
        {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_loss": float(best_val_loss),
        },
        str(path),
    )


def train_teacher(logger):
    seed_everything(DEFAULT_SEED)

    device = _select_device()
    use_amp = device.type == "cuda"
    amp_dtype = tu.get_amp_dtype(cfg)
    logger.info(f"Device: {device} | AMP: {use_amp} | amp_dtype: {amp_dtype}")

    if device.type == "cuda" and cfg.get("allow_tf32", True):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    run_dir = Path(cfg["run_dir"])
    ckpt_dir = run_dir / cfg["ckpt_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    teacher_filename = cfg.get("teacher_filename", "frozen_teacher.pth")
    best_model_path = ckpt_dir / teacher_filename

    dataset = MRCTDataset(
        root=cfg["root"],
        modality=cfg["modality"],
        use_preproc_files=cfg.get("use_preproc_files", False),
        use_pruned_data=cfg.get("use_pruned_data", False),
        patch_size=cfg["patch_size"],
        anatomy=cfg.get("anatomy", None),
        num_adjacent_slices=cfg["num_adjacent_slices"],
        foreground_threshold=cfg["foreground_threshold"],
        cache_size=cfg["cache_size"],
        transform=_teacher_transform,
    )

    logger.info(f"Teacher dataset size: {len(dataset)}")
    train_loader, val_loader, train_ds, val_ds, train_patients, val_patients = build_dataloaders(
        dataset=dataset,
        cfg=cfg,
        seed=DEFAULT_SEED,
        device=device,
    )
    logger.info(f"Patients: {len(train_patients)} train | {len(val_patients)} val")
    logger.info(f"Samples: {len(train_ds)} train | {len(val_ds)} val")

    model = _build_teacher_unet(dataset, cfg, device)
    loss_fn = DiceLoss(sigmoid=True)
    optimizer = _build_optimizer(model, device, lr=float(cfg["lr"]))
    scheduler = tu.build_scheduler(optimizer, int(cfg.get("scheduler_patience", 10)))

    scaler_enabled = use_amp and amp_dtype == torch.float16
    grad_scaler_cls = getattr(getattr(torch, "amp", None), "GradScaler", torch.cuda.amp.GradScaler)
    scaler = grad_scaler_cls(enabled=scaler_enabled)

    start_epoch, best_val_loss = _load_resume_if_available(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        resume_path=cfg.get("resume_path"),
        device=device,
    )

    epochs = int(cfg["epochs"])
    early_stop_patience = int(cfg.get("teacher_patience", cfg.get("scheduler_patience", 10)))
    no_improve = 0

    logger.info(f"Starting teacher training from epoch {start_epoch} to {epochs}")
    
    # Time tracking
    training_start_time = time.time()
    epoch_times = []
    
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
    
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            inputs, labels = _maybe_channels_last(inputs, labels, dataset.is_3d, cfg, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(inputs)
                loss = loss_fn(logits, labels)

            if not torch.isfinite(loss):
                logger.warning("Non-finite teacher loss encountered; skipping step")
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += float(loss.item())
            train_steps += 1
            
            # Log progress every 10% of batches
            if (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
                logger.info(f"Epoch {epoch}/{epochs} - Train batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        train_loss = train_loss / max(1, train_steps)

        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                inputs = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                inputs, labels = _maybe_channels_last(inputs, labels, dataset.is_3d, cfg, device)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    logits = model(inputs)
                    loss = loss_fn(logits, labels)

                val_loss += float(loss.item())
                val_steps += 1

        val_loss = val_loss / max(1, val_steps)
        scheduler.step(val_loss)

        # Calculate time metrics
        epoch_duration = time.time() - epoch_start_time
        epoch_times.append(epoch_duration)
        elapsed_time = time.time() - training_start_time
        
        # Estimate time remaining based on average epoch time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        epochs_remaining = epochs - epoch
        estimated_time_remaining = avg_epoch_time * epochs_remaining

        logger.info(
            f"Epoch {epoch:03d} | "
            f"train_dice_loss={train_loss:.4f} | "
            f"val_dice_loss={val_loss:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.1e}"
        )
        logger.info(
            f"Time - Epoch: {format_time(epoch_duration)} | "
            f"Elapsed: {format_time(elapsed_time)} | "
            f"Remaining: {format_time(estimated_time_remaining)} (est.)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            _save_teacher_checkpoint(best_model_path, epoch, model, optimizer, scheduler, scaler, best_val_loss)
            logger.info(f"Saved best teacher checkpoint: {best_model_path}")
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            logger.info(f"Early stopping teacher training (patience={early_stop_patience})")
            break

    total_training_time = time.time() - training_start_time
    logger.info(f"Teacher training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Total training time: {format_time(total_training_time)}")
    return best_model_path


if __name__ == "__main__":
    # Parse command-line arguments
    args = _parse_args()
    
    # Override config with arguments
    _override_cfg_with_args(cfg, args)
    
    # Set up logging
    run_dir = Path(cfg["run_dir"])
    logger = _setup_logging(run_dir, args.log_file)
    
    logger.info("=" * 60)
    logger.info("Starting teacher segmentor training")
    logger.info("=" * 60)
    logger.info(f"Configuration after overrides:")
    logger.info(f"  run_dir: {cfg['run_dir']}")
    logger.info(f"  patch_size: {cfg['patch_size']}")
    logger.info(f"  num_adjacent_slices: {cfg['num_adjacent_slices']}")
    logger.info(f"  use_pruned_data: {cfg['use_pruned_data']}")
    logger.info(f"  use_preproc_files: {cfg['use_preproc_files']}")
    logger.info(f"  epochs: {cfg['epochs']}")
    logger.info(f"  log_file: {args.log_file}")
    logger.info("=" * 60)
    
    train_teacher(logger)

