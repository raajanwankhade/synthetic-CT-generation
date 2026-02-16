import os
import io
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

# from torchmetrics.functional import structural_similarity_index_measure as ssim
# from torchmetrics.functional import peak_signal_noise_ratio as psnr

SHUTDOWN_REQUESTED = False


def init_wandb(cfg: dict, seed: int, use_amp: bool, is_3d: bool):
    os.environ["WANDB_MODE"] = cfg.get("wandb_mode", "online")
    wandb.login()
    return wandb.init(
        project=cfg["wandb_project"],
        name=cfg["wandb_run_name"],
        config={
            "seed": seed,
            "epochs": cfg["epochs"],
            "batch_size": cfg["batch_size"],
            "lr": cfg["lr"],
            "patch_size": cfg["patch_size"],
            "anatomy": cfg.get("anatomy", None),
            "num_adjacent_slices": cfg["num_adjacent_slices"],
            "model": "ResNet3D_EncoderDecoder" if is_3d else "ResNet2D_EncoderDecoder",
            "scheduler": "ReduceLROnPlateau",
            "loss": "L1 + SSIM + Seg Loss",
            "amp": use_amp,
        },
        mode=cfg["wandb_mode"],
        resume=cfg["resume_wandb"],
        id=cfg["id_wandb"],
    )


# @torch.no_grad()
# def compute_metrics(pred, target, ct_range=(-1024, 3071)):
#     min_hu, max_hu = ct_range
#     data_range = max_hu - min_hu

#     pred_norm = (pred - min_hu) / data_range
#     target_norm = (target - min_hu) / data_range

#     pred_norm = torch.clamp(pred_norm, 0, 1)
#     target_norm = torch.clamp(target_norm, 0, 1)

#     ssim_val = ssim(pred_norm, target_norm, data_range=1.0).item()
#     psnr_val = psnr(pred_norm, target_norm, data_range=1.0).item()
#     return ssim_val, psnr_val


@torch.no_grad()
def visualize_batch(encoder, decoder, dataloader, device=None, num_samples=4):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval()
    decoder.eval()

    batch = next(iter(dataloader))
    x = batch["input"].to(device)
    y = batch["label"].to(device)

    feats = encoder(x)
    preds = decoder(feats)

    x = x.cpu()
    y = y.cpu()
    preds = preds.cpu()

    n = min(num_samples, x.size(0))

    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(10, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        if x.dim() == 5:
            mid = x.shape[2] // 2
            inp = x[i, 0, mid].numpy()
            gt = y[i, 0, mid].numpy()
            gen = preds[i, 0, mid].numpy()
        else:
            inp = x[i, x.shape[1] // 2].numpy() if x.shape[1] > 1 else x[i, 0].numpy()
            gt = y[i, 0].numpy()
            gen = preds[i, 0].numpy()

        for j, img in enumerate([inp, gt, gen]):
            vmin, vmax = np.percentile(img, [2, 98])
            axes[i, j].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            axes[i, j].axis("off")

        axes[i, 0].set_title("Input")
        axes[i, 1].set_title("Ground Truth (CT)")
        axes[i, 2].set_title("Generated")

    plt.tight_layout()
    return fig


class CheckpointManager:
    def __init__(self, save_dir="checkpoints", monitor="val_loss", mode="min", filename="best.pth"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.best = float("inf") if mode == "min" else -float("inf")
        self.monitor = monitor
        self.mode = mode
        self.filename = filename
        self.best_path = None
        self.best_epoch = None
        self.best_metrics = None

    def update(self, epoch, encoder, decoder, optimizer, scheduler, scaler, val_metrics):
        if self.monitor not in val_metrics:
            raise KeyError(f"Monitor key '{self.monitor}' not in val_metrics: {list(val_metrics.keys())}")
        score = float(val_metrics[self.monitor])
        improved = (score < self.best) if self.mode == "min" else (score > self.best)
        if improved:
            self.best = score
            self.best_epoch = int(epoch)
            self.best_metrics = dict(val_metrics)
            ckpt_path = os.path.join(self.save_dir, self.filename)
            self.save_epoch(ckpt_path, epoch, encoder, decoder, optimizer, scheduler, scaler)
            self.best_path = ckpt_path
            print(f"✅ Saved BEST checkpoint: epoch={epoch} {self.monitor}={score:.5f} -> {ckpt_path}")

    def save_epoch(self, ckpt_path, epoch, encoder, decoder, optimizer, scheduler, scaler):
        torch.save({
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }, ckpt_path)

    def load_best(self, encoder, decoder, optimizer=None, device=None):
        if not self.best_path:
            candidate = os.path.join(self.save_dir, self.filename)
            if os.path.exists(candidate):
                self.best_path = candidate
            else:
                raise ValueError("No best checkpoint found.")
        ckpt = torch.load(self.best_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        epoch = ckpt.get("epoch", None)
        print(f"🔁 Loaded best checkpoint from {self.best_path} (epoch={epoch})")
        return ckpt


def get_amp_dtype(cfg: dict) -> torch.dtype:
    amp_dtype_name = str(cfg.get("amp_dtype", "bf16")).lower()
    return torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16


def build_optimizer(encoder: nn.Module, decoder: nn.Module, device: torch.device, lr: float, verbose: bool = True):
    try:
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=lr,
            fused=(device.type == "cuda"),
        )
        if device.type == "cuda" and verbose:
            print("✅ Using AdamW(fused=True) on CUDA")
        return optimizer
    except TypeError:
        return torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            weight_decay=0,
            lr=lr,
        )


def build_scheduler(optimizer, scheduler_patience: int):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience
    )


# def maybe_compile_models(encoder: nn.Module, decoder: nn.Module, cfg: dict):
#     compile_enabled = False
#     if cfg.get("torch_compile", False):
#         try:
#             slurm_tmp = os.environ.get("SLURM_TMPDIR")
#             if slurm_tmp:
#                 os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(slurm_tmp, "torchinductor"))
#             try:
#                 from torch._dynamo import config as dynamo_config  # type: ignore
#                 dynamo_config.suppress_errors = True
#             except Exception:
#                 pass

#             mode = cfg.get("torch_compile_mode", "reduce-overhead")
#             encoder = cast(nn.Module, torch.compile(encoder, mode=mode))
#             decoder = cast(nn.Module, torch.compile(decoder, mode=mode))
#             compile_enabled = True
#             print(f"✅ torch.compile enabled (mode={mode})")
#         except Exception as e:
#             print(f"⚠️ torch.compile not enabled: {e}")
#     return encoder, decoder, compile_enabled


def enforce_channels_last(x: torch.Tensor, y: torch.Tensor, m: torch.Tensor, device: torch.device):
    if device.type != "cuda":
        return x, y, m
    if x.dim() == 5:  # 3D
        return (
            x.contiguous(memory_format=torch.channels_last_3d),
            y.contiguous(memory_format=torch.channels_last_3d),
            m.contiguous(memory_format=torch.channels_last_3d),
        )
    return (
        x.contiguous(memory_format=torch.channels_last),
        y.contiguous(memory_format=torch.channels_last),
        m.contiguous(memory_format=torch.channels_last),
    )


# def ssim_loss(preds: torch.Tensor, y_norm: torch.Tensor, ssim_down: int) -> torch.Tensor:
#     # preds_norm = torch.clamp((preds_hu + 1024) / 4095, 0, 1)
#     # y_norm = torch.clamp((y_hu + 1024) / 4095, 0, 1)
#     preds_norm = torch.clamp(preds, 0, 1)
#     if ssim_down > 1:
#         if preds_norm.dim() == 5:
#             preds_norm = F.avg_pool3d(preds_norm, kernel_size=ssim_down)
#             y_norm = F.avg_pool3d(y_norm, kernel_size=ssim_down)
#         else:
#             preds_norm = F.avg_pool2d(preds_norm, kernel_size=ssim_down)
#             y_norm = F.avg_pool2d(y_norm, kernel_size=ssim_down)
#     # torchmetrics SSIM can return values in [-1, 1] or exceed bounds due to numerical issues
#     # Clamp to [0, 1] to ensure loss is in valid [0, 1] range
#     ssim_val = torch.clamp(ssim(preds_norm, y_norm, data_range=1.0), 0.0, 1.0)
#     return 1 - ssim_val
