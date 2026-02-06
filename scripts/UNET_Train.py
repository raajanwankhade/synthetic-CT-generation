import os
from utils.env import load_and_setup_env

load_and_setup_env("secrets.json", set_kaggle=True, set_wandb=True)

# General
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm.auto import tqdm
import io
from PIL import Image

import wandb
from config import cfg

# Model architecture
from models import build_model
from training.wandb_utils import init_wandb
from training import callbacks
from training.visualization import visualize_batch, plot_history
from training.trainer import train_model

# Shutdown flag
SHUTDOWN_FLAG_PATH = os.path.join(cfg.paths.run_dir, "stop.flag")
# Reproducibility + device helpers
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism vs speed (HPC training typically prefers speed)
    torch.backends.cudnn.deterministic = bool(cfg.perf.deterministic)
    torch.backends.cudnn.benchmark = bool(cfg.perf.cudnn_benchmark)

SEED = 42
seed_everything(SEED)

# -------------------------------------------------------------------------
# Device selection (rank-aware) + AMP safety
# -------------------------------------------------------------------------
def _select_device():
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    return torch.device("cpu")

device = _select_device()
use_amp = (device.type == "cuda")
print("Device:", device, "| AMP:", use_amp, "| CUDA devices:", torch.cuda.device_count())

# H100-friendly math (TF32 for float32 matmuls/convs)
if device.type == "cuda" and cfg.perf.allow_tf32:
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
root = cfg.paths.root
MAX_EPOCHS = cfg.training.epochs
IS_3D = str(cfg.data.num_adjacent_slices).lower() == "3d"
# Faster matmul on newer torch (safe to ignore if unavailable)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Training
from data.dataset import MRCTDataset
from data.loaders import build_dataloaders

#Run dir
run_dir = Path(cfg.paths.run_dir)
init_wandb(cfg, SEED, use_amp, IS_3D)

dataset = MRCTDataset(
    root=root,
    modality=cfg.data.modality,
    patch_size=None if IS_3D else cfg.data.patch_size,
    anatomy=cfg.data.anatomy,
    num_adjacent_slices=cfg.data.num_adjacent_slices,
    max_retries=20,
    foreground_threshold=cfg.data.foreground_threshold,
    cache_size=cfg.data.cache_size,  # per-worker LRU
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

# Build model using factory
model = build_model(is_3d=IS_3D)


trained_model, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    start_epoch=cfg.training.start,
    num_epochs=MAX_EPOCHS,
    lr=cfg.training.lr,
    device=device,
    wandb_log=(cfg.logging.wandb_mode != "disabled"),
    lambda_ssim=cfg.training.lambda_ssim,
    scheduler_patience=cfg.training.scheduler_patience,
    grad_clip=cfg.training.grad_clip,
    sanity_steps=cfg.training.sanity_steps,
    resume_path=cfg.training.resume_path,
    cfg=cfg,
    run_dir=run_dir,
    shutdown_flag_path=SHUTDOWN_FLAG_PATH,
)

plot_history(history, str(run_dir / cfg.logging.loss_plot_name))

wandb.finish()
visualize_batch(trained_model, val_loader)

if callbacks.SHUTDOWN_REQUESTED:
    print("[shutdown] Shutdown complete. Removing flag file.")
    os.remove(SHUTDOWN_FLAG_PATH)
