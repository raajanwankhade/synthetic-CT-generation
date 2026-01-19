import os
import json
secrets_path = 'secrets.json'
with open(secrets_path, 'r') as f:
    secrets = json.load(f)
os.environ["WANDB_MODE"] = "online"
os.environ["KAGGLE_USERNAME"] = secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = secrets["KAGGLE_KEY"]
os.environ["WANDB_API_KEY"] = secrets["wandb_API_KEY"]
# General
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

#For loss
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
# from piq import ssim, psnr, vif_p

# For logging
import os
import io
import wandb
from PIL import Image
from config import cfg

# Reproducibility + device helpers
import random

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

# H100-friendly math (TF32 for float32 matmuls/convs)
if torch.cuda.is_available() and cfg.get("allow_tf32", True):
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
print("Device:", device, "| AMP:", use_amp)
root = cfg["root"]
MAX_EPOCHS = cfg["epochs"]
# Faster matmul on newer torch (safe to ignore if unavailable)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Training
from dataset_class import MRCTDataset
from torch.utils.data import DataLoader, Subset

# Login will use WANDB_API_KEY from environment if present.
# Tip (Kaggle): add it in "Secrets" instead of writing it here.
wandb.login()

#Run dir
run_dir = Path(cfg["run_dir"])
# Networks

class ResBlock(nn.Module):
    """Simple residual block: conv -> norm -> ReLU -> conv -> norm + skip"""
    def __init__(self, channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = norm_layer(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_layer(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResEncoder2D(nn.Module):
    """
    Fixed version: returns one feature per true downsampling level.
    """
    def __init__(self, in_ch=1, base_ch=32, num_stages=4, blocks_per_stage=2, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=7, stride=1, padding=3, bias=False),
            norm_layer(base_ch),
            nn.ReLU(inplace=True)
        )

        stages = []
        ch = base_ch
        for s in range(num_stages):
            blocks = [ResBlock(ch, norm_layer=norm_layer) for _ in range(blocks_per_stage)]
            stage = nn.Sequential(*blocks)
            stages.append(stage)
            if s != num_stages - 1:
                down = nn.Sequential(
                    nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    norm_layer(ch * 2),
                    nn.ReLU(inplace=True)
                )
                stages.append(down)
                ch *= 2
        self.stages = nn.ModuleList(stages)
        self.out_ch = ch

    def forward(self, x):
        feats = []
        x = self.stem(x)
        for m in self.stages:
            x = m(x)
            # only save feature maps that change spatial resolution (avoid duplicate same-size)
            if len(feats) == 0 or feats[-1].shape[-1] != x.shape[-1]:
                feats.append(x)
        # should produce features roughly [256,128,64,32]
        return feats


def test_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = ResEncoder2D(in_ch=1, base_ch=32, num_stages=4, blocks_per_stage=2).to(device)
    x = torch.randn(4, 1, 256, 256, device=device)  # batch of input patches
    feats = enc(x)
    print("Number of returned feature maps:", len(feats))
    for i, f in enumerate(feats):
        print(f"feat[{i}] shape: {tuple(f.shape)}")
    # expected (approx): feat[0] (4,32,256,256), feat[1] (4,64,128,128), feat[2] (4,128,64,64), feat[3] (4,256,32,32)
    return feats

# test_encoder()

class UpConvBlock(nn.Module):
    """Upsample by factor 2 (once) + conv + residual blocks."""
    def __init__(self, in_ch, out_ch, blocks=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True)
        ]
        for _ in range(blocks):
            layers.append(ResBlock(out_ch, norm_layer=norm_layer))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ResDecoder2D(nn.Module):
    """
    Decoder that takes encoder features (list) and reconstructs an image
    with same spatial resolution as feats[0].
    """
    def __init__(self, encoder_feature_channels, out_ch=1, blocks_per_up=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.feat_chs = encoder_feature_channels
        self.up_blocks = nn.ModuleList()

        for i in range(len(encoder_feature_channels) - 1, 0, -1):
            in_ch = encoder_feature_channels[i]
            skip_ch = encoder_feature_channels[i - 1]
            self.up_blocks.append(
                UpConvBlock(in_ch + skip_ch, skip_ch, blocks=blocks_per_up, norm_layer=norm_layer)
            )

        final_in_ch = encoder_feature_channels[0]
        self.final_conv = nn.Sequential(
            nn.Conv2d(final_in_ch, final_in_ch // 2, kernel_size=3, padding=1, bias=False),
            norm_layer(final_in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_in_ch // 2, out_ch, kernel_size=1)
        )

    def forward(self, feats):
        x = feats[-1]  # start from deepest (lowest res)
        for i, up in enumerate(self.up_blocks):
            skip_idx = len(feats) - 2 - i
            skip = feats[skip_idx]
            # only one upsample per stage
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = up(x)
        out = self.final_conv(x)
        return out

def test_encoder_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = ResEncoder2D(in_ch=1, base_ch=32, num_stages=4, blocks_per_stage=2).to(device)
    dummy = torch.randn(1, 1, 256, 256, device=device)
    feats = enc(dummy)
    feat_chs = [f.shape[1] for f in feats]
    print("Encoder feature channels:", feat_chs)
    for i, f in enumerate(feats):
        print(f"feat[{i}] shape:", f.shape)
    dec = ResDecoder2D(encoder_feature_channels=feat_chs, out_ch=1).to(device)
    out = dec(enc(torch.randn(4, 1, 256, 256, device=device)))
    print("Output shape:", out.shape)


# Loss

# class PerceptualLoss(nn.Module):
#     def __init__(self, layer='relu3_3', device='cuda'):
#         super().__init__()
#         vgg = models.vgg19(pretrained=True).features.eval()
#         self.slice = nn.Sequential(*[vgg[x] for x in range(16)])  # up to relu3_3
#         for p in self.slice.parameters():
#             p.requires_grad = False
#         self.slice = self.slice.to(device)  # <-- move to GPU
#         self.criterion = nn.L1Loss()
#         self.device = device

#     def forward(self, x, y):
#         # Ensure inputs are float32 and on correct device
#         x = x.to(self.device, dtype=torch.float32)
#         y = y.to(self.device, dtype=torch.float32)

#         # Convert grayscale → RGB if needed
#         if x.shape[1] == 1:
#             x = x.repeat(1, 3, 1, 1)
#             y = y.repeat(1, 3, 1, 1)

#         with torch.no_grad():
#             x_feat = self.slice(x)
#             y_feat = self.slice(y)

#         # Compute perceptual (feature-space) distance
#         loss = self.criterion(x_feat, y_feat)
#         return loss

# class InformativeLoss(nn.Module):
#     def __init__(self, w_ssim=1.0, w_psnr=0.5, w_vif=1.0, w_perc=1.0, device='cuda'):
#         super().__init__()
#         self.w_ssim = w_ssim
#         self.w_psnr = w_psnr
#         # self.w_vif = w_vif
#         self.w_perc = w_perc
#         self.device = device
#         self.perc_loss = PerceptualLoss(device=device)

#     def forward(self, pred, target):
#         # Normalize range for perceptual + PIQ metrics
#         pred = torch.clamp(pred, 0.0, 1.0)
#         target = torch.clamp(target, 0.0, 1.0)

#         ssim_val = ssim(pred, target)
#         psnr_val = psnr(pred, target)
#         # vif_val = torch.clamp(vif_p(pred, target, data_range=1.0), min=0.0, max=1.0)

#         ssim_loss = 1 - ssim_val
#         psnr_loss = 1 / (psnr_val + 1e-8)
#         # vif_loss = 1 - vif_val
#         perc_loss = self.perc_loss(pred, target)

#         total = (
#             self.w_ssim * ssim_loss +
#             self.w_psnr * psnr_loss +
#             # self.w_vif * vif_loss +
#             self.w_perc * perc_loss
#         )

#         if not torch.isfinite(total):
#             print("⚠️ NaN detected in loss components:")
#             print(f"  SSIM={ssim_val.item()}, PSNR={psnr_val.item()}, PERC={perc_loss.item()}")
#             total = torch.tensor(0.0, device=pred.device, requires_grad=True)

#         return total, {
#             "SSIM": ssim_val.item(),
#             "PSNR": psnr_val.item(),
#             # "VIF": vif_val.item(),
#             "Perceptual": perc_loss.detach().item(),
#         }


# def test_combined_loss(criterion, device='cuda', img_size=(1, 1, 128, 128)):
#     """
#     Quick functional test for combined loss class.

#     Args:
#         criterion: instance of your loss class (e.g., InformativeLoss)
#         device: 'cuda' or 'cpu'
#         img_size: (batch, channels, H, W)
#     """
#     print(f"🔍 Testing Combined Loss on device: {device}")

#     # Create dummy data
#     pred = torch.rand(img_size, device=device)
#     target = torch.rand(img_size, device=device)

#     # Forward pass
#     with torch.no_grad():
#         total_loss, metrics = criterion(pred, target)

#     # Print results
#     print(f"\n✅ Total Loss: {total_loss.item():.6f}")
#     print("📊 Sub-metrics:")
#     for k, v in metrics.items():
#         print(f"   {k:>10s}: {v:.6f}")


@torch.no_grad()
def compute_metrics(pred, target, ct_range=(-1024, 3071)):
    """
    Computes SSIM and PSNR between predicted and ground truth CT volumes.

    Args:
        pred (torch.Tensor): predicted CT image (B, 1, H, W)
        target (torch.Tensor): ground-truth CT image (B, 1, H, W)
        ct_range (tuple): (min_HU, max_HU) for CT normalization
    """
    min_hu, max_hu = ct_range
    data_range = max_hu - min_hu

    # Normalize to [0, 1] range
    pred_norm = (pred - min_hu) / data_range
    target_norm = (target - min_hu) / data_range

    pred_norm = torch.clamp(pred_norm, 0, 1)
    target_norm = torch.clamp(target_norm, 0, 1)

    ssim_val = ssim(pred_norm, target_norm, data_range=1.0).item()
    psnr_val = psnr(pred_norm, target_norm, data_range=1.0).item()
    return ssim_val, psnr_val

# Logging and Checkpointing

# Initialize W&B (set WANDB_MODE=offline or disabled as needed)
wandb.init(
    project=cfg["wandb_project"],
    name=cfg["wandb_run_name"],
    config={
        "seed": SEED,
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "patch_size": cfg["patch_size"],
        "model": "ResNet2D_EncoderDecoder",
        "scheduler": "ReduceLROnPlateau",
        "loss": "L1 + SSIM",
        "amp": use_amp,
    },
    mode=cfg["wandb_mode"],
 )

@torch.no_grad()
def visualize_batch(encoder, decoder, dataloader, device=None, num_samples=4):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval()
    decoder.eval()

    # Get one batch
    batch = next(iter(dataloader))
    x = batch["input"].to(device)
    y = batch["label"].to(device)

    # Forward pass
    feats = encoder(x)
    preds = decoder(feats)

    # Move to CPU and detach
    x = x.cpu()
    y = y.cpu()
    preds = preds.cpu()

    # Select limited samples to display
    n = min(num_samples, x.size(0))

    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(10, 3*n))
    if n == 1:
        axes = axes[None, :]  # handle single row case

    for i in range(n):
        inp = x[i, x.shape[1]//2].numpy() if x.shape[1] > 1 else x[i,0].numpy()
        gt = y[i,0].numpy()
        gen = preds[i,0].numpy()

        # Apply simple normalization for display
        for j, img in enumerate([inp, gt, gen]):
            vmin, vmax = np.percentile(img, [2, 98])
            axes[i, j].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            axes[i, j].axis("off")

        axes[i, 0].set_title("Input")
        axes[i, 1].set_title("Ground Truth (CT)")
        axes[i, 2].set_title("Generated")

    plt.tight_layout()
    return fig

# -------------------------------------------------------------------------
# Checkpoint Manager
# -------------------------------------------------------------------------
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
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "val_metrics": val_metrics,
            }, ckpt_path)
            self.best_path = ckpt_path
            print(f"✅ Saved BEST checkpoint: epoch={epoch} {self.monitor}={score:.5f} -> {ckpt_path}")

    def load_best(self, encoder, decoder, optimizer=None, device=None):
        if not self.best_path:
            # Fallback: allow loading best.pth even if update() wasn't called in this session
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

# Training

# # -------------------------------------------------------------------------
# # Utility: Compute metrics
# # -------------------------------------------------------------------------
# @torch.no_grad()
# def compute_metrics(pred, target):
#     pred = torch.clamp(pred, 0, 1)
#     target = torch.clamp(target, 0, 1)
#     return ssim(pred, target).item(), psnr(pred, target).item()

# -------------------------------------------------------------------------
# Training function (W&B integrated)
# -------------------------------------------------------------------------
def train_model(
    encoder,
    decoder,
    train_loader,
    val_loader,
    num_epochs=100,
    start_epoch=1,
    lr=1e-4,
    val_split=0.2,
    save_dir="checkpoints",
    scheduler_patience=5,
    device=None,
    wandb_log=True,
    lambda_ssim=1000.0,
    resume_path=None,
    grad_clip=None,
    sanity_steps=1,
 ):

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    amp_dtype_name = str(cfg.get("amp_dtype", "bf16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16

    # Move models
    encoder.to(device)
    decoder.to(device)

    # channels_last helps conv throughput on recent GPUs
    if cfg.get("channels_last", True) and device.type == "cuda":
        encoder = encoder.to(memory_format=torch.channels_last)
        decoder = decoder.to(memory_format=torch.channels_last)

    # torch.compile (PyTorch 2.x)
    if cfg.get("torch_compile", False):
        try:
            mode = cfg.get("torch_compile_mode", "max-autotune")
            encoder = torch.compile(encoder, mode=mode)
            decoder = torch.compile(decoder, mode=mode)
            print(f"✅ torch.compile enabled (mode={mode})")
        except Exception as e:
            print(f"⚠️ torch.compile not enabled: {e}")

    criterion = nn.L1Loss()

    # Prefer fused optimizer on CUDA if available
    try:
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=lr,
            fused=(device.type == "cuda"),
        )
        if device.type == "cuda":
            print("✅ Using AdamW(fused=True) on CUDA")
    except TypeError:
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience, verbose=True
    )
    ckpt_mgr = CheckpointManager(save_dir, monitor="val_loss", mode="min", filename=cfg["ckpt_filename"])

    # GradScaler is only needed for FP16; BF16 generally doesn't use scaling
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    ssim_down = int(cfg.get("ssim_downsample", 1))

    def _ssim_loss(preds_hu: torch.Tensor, y_hu: torch.Tensor) -> torch.Tensor:
        preds_norm = torch.clamp((preds_hu + 1024) / 4095, 0, 1)
        y_norm = torch.clamp((y_hu + 1024) / 4095, 0, 1)
        if ssim_down > 1:
            preds_norm = F.avg_pool2d(preds_norm, kernel_size=ssim_down)
            y_norm = F.avg_pool2d(y_norm, kernel_size=ssim_down)
        return 1 - ssim(preds_norm, y_norm, data_range=1.0)

    # -------------------------------------------------------------
    # Resume if checkpoint provided
    # -------------------------------------------------------------
    if resume_path is not None:
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
    else:
        print("🆕 Starting fresh training")

    # -------------------------------------------------------------
    # Sanity check: run a couple train steps to catch shape/NaN issues early
    # -------------------------------------------------------------
    if sanity_steps and sanity_steps > 0:
        encoder.train()
        decoder.train()
        it = iter(train_loader)
        for _ in range(int(sanity_steps)):
            batch = next(it)
            x = batch["input"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            if cfg.get("channels_last", True) and device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
                y = y.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                feats = encoder(x)
                preds = decoder(feats)
                loss_l1 = criterion(preds, y)
                loss_ssim = _ssim_loss(preds, y)
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

    history = {"train_loss": [], "val_loss": [], "SSIM": [], "PSNR": []}

    for epoch in tqdm(range(start_epoch, num_epochs + 1), desc="Progress"):
        print(f"\n🧾 Epoch {epoch}/{num_epochs}")
        encoder.train()
        decoder.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            x = batch["input"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            if cfg.get("channels_last", True) and device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
                y = y.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                feats = encoder(x)
                preds = decoder(feats)
                loss_l1 = criterion(preds, y)

                loss_ssim = _ssim_loss(preds, y)
                loss = loss_l1 + lambda_ssim * loss_ssim

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
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= max(1, len(train_loader))

        # ---------------------- VALIDATION ----------------------
        encoder.eval()
        decoder.eval()
        val_loss, ssim_sum, psnr_sum = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                x = batch["input"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)

                if cfg.get("channels_last", True) and device.type == "cuda":
                    x = x.contiguous(memory_format=torch.channels_last)
                    y = y.contiguous(memory_format=torch.channels_last)

                feats = encoder(x)
                preds = decoder(feats)

                loss_l1 = criterion(preds, y)
                loss_ssim = _ssim_loss(preds, y)
                loss = loss_l1 + lambda_ssim * loss_ssim

                val_loss += float(loss.item())
                s, p = compute_metrics(preds, y)
                ssim_sum += s
                psnr_sum += p

        val_loss /= max(1, len(val_loader))
        ssim_avg = ssim_sum / max(1, len(val_loader))
        psnr_avg = psnr_sum / max(1, len(val_loader))

        val_metrics = {"val_loss": val_loss, "SSIM": ssim_avg, "PSNR": psnr_avg}
        scheduler.step(val_loss)
        ckpt_mgr.update(epoch, encoder, decoder, optimizer, scheduler, scaler, val_metrics)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"SSIM: {ssim_avg:.4f} | PSNR: {psnr_avg:.2f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e}"
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["SSIM"].append(ssim_avg)
        history["PSNR"].append(psnr_avg)

        if wandb_log:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "SSIM": ssim_avg,
                "PSNR": psnr_avg,
                "lr": optimizer.param_groups[0]['lr'],
            })

            log_every = int(cfg.get("log_images_every", 0))
            save_every = int(cfg.get("save_vis_every", 0))

            if log_every > 0 and (epoch % log_every == 0):
                fig = visualize_batch(encoder, decoder, val_loader, device=device, num_samples=4)
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                image = Image.open(buf)
                wandb.log({f"epoch_{epoch}_output": wandb.Image(image, caption=f"Epoch {epoch}")})
                plt.close(fig)

            if save_every > 0 and (epoch % save_every == 0):
                fig = visualize_batch(encoder, decoder, val_loader, device=device, num_samples=4)
                fig.savefig(run_dir / cfg["Visualisation_Fig"])
                plt.close(fig)

    print("\n🏁 Training complete!")
    ckpt_mgr.load_best(encoder, decoder, optimizer, device)
    return encoder, decoder, history

# your dataset instance
# dataset = MyDataset(data_dir="/path/to/patches")



dataset_2d = MRCTDataset(
    root=root,
    modality=cfg["modality"],
    patch_size=cfg["patch_size"],
    num_adjacent_slices=cfg["num_adjacent_slices"],  # pure 2D
    max_retries=20,
    foreground_threshold=cfg["foreground_threshold"],
    cache_size=cfg["cache_size"],  # per-worker LRU
 )
print("Dataset size:", len(dataset_2d))
sample = dataset_2d[0]
print(sample["input"].shape, sample["label"].shape, sample["mask"].shape)
assert sample["input"].shape[0] == 1, "Expected 2D input with 1 channel"

# device/use_amp already defined in the reproducibility cell above
print(device)

val_split = cfg["val_split"]

# Patient-level split to avoid leakage (train/val should not share patients)
patient_ids = np.array([p.name for (p, _) in dataset_2d.samples])
uniq_patients = np.unique(patient_ids)
rng = np.random.default_rng(SEED)
rng.shuffle(uniq_patients)

n_val_patients = int(len(uniq_patients) * val_split)
val_patients = set(uniq_patients[:n_val_patients])
train_patients = set(uniq_patients[n_val_patients:])

train_idx = np.where(np.isin(patient_ids, list(train_patients)))[0].tolist()
val_idx   = np.where(np.isin(patient_ids, list(val_patients)))[0].tolist()

from torch.utils.data import Subset
train_ds = Subset(dataset_2d, train_idx)
val_ds   = Subset(dataset_2d, val_idx)

print("Patients:", len(train_patients), "train |", len(val_patients), "val")
print("Samples:", len(train_ds), "train |", len(val_ds), "val")

# DataLoader: robust defaults across CPU/GPU + reproducible worker seeds
def _seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

pin = (device.type == "cuda")
train_loader = DataLoader(
    train_ds,
    batch_size=cfg["batch_size"],
    shuffle=True,
    num_workers=cfg["num_workers_train"],
    pin_memory=pin,
    persistent_workers=(cfg["num_workers_train"] > 0),
    worker_init_fn=_seed_worker,
    generator=g,
    prefetch_factor=cfg["prefetch_factor"],
 )

val_loader = DataLoader(
    val_ds,
    batch_size=cfg["batch_size"],
    shuffle=False,
    num_workers=cfg["num_workers_val"],
    pin_memory=pin,
    persistent_workers=(cfg["num_workers_val"] > 0),
    worker_init_fn=_seed_worker,
    generator=g,
    prefetch_factor=cfg["prefetch_factor"],
 )

# Build model
enc = ResEncoder2D(in_ch=1, base_ch=32, num_stages=4, blocks_per_stage=2)
with torch.no_grad():
    feats = enc(torch.randn(1, 1, 256, 256))
feat_chs = [f.shape[1] for f in feats]
dec = ResDecoder2D(encoder_feature_channels=feat_chs, out_ch=1)


trained_enc, trained_dec, history = train_model(
    encoder=enc,
    decoder=dec,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=MAX_EPOCHS,
    lr=cfg["lr"],
    device=device,
    wandb_log=True,
    lambda_ssim=cfg["lambda_ssim"],
    scheduler_patience=cfg["scheduler_patience"],
    grad_clip=cfg["grad_clip"],
    sanity_steps=cfg["sanity_steps"],
    save_dir=run_dir / cfg["ckpt_dir"],
)

# --- Plot each training metric separately ---
epochs = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(14, 18))

# 1. Training & Validation Loss
plt.subplot(3, 1, 1)
plt.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
plt.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Composite Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# 2. SSIM
plt.subplot(3, 1, 2)
plt.plot(epochs, history["SSIM"], label="SSIM", color='tab:blue', linewidth=2)
plt.title("SSIM over Epochs")
plt.xlabel("Epoch")
plt.ylabel("SSIM")
plt.legend()
plt.grid(True, alpha=0.3)

# 3. PSNR
plt.subplot(3, 1, 3)
plt.plot(epochs, history["PSNR"], label="PSNR", color='tab:orange', linewidth=2)
plt.title("PSNR over Epochs")
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.legend()
plt.grid(True, alpha=0.3)

# 4. VIF
# plt.subplot(5, 1, 4)
# plt.plot(epochs, history["VIF"], label="VIF", color='tab:green', linewidth=2)
# plt.title("Visual Information Fidelity (VIF)")
# plt.xlabel("Epoch")
# plt.ylabel("VIF")
# plt.legend()
# plt.grid(True, alpha=0.3)

# # 5. Perceptual Loss
# plt.subplot(4, 1, 4)
# plt.plot(epochs, history["Perceptual"], label="Perceptual Loss", color='tab:red', linewidth=2)
# plt.title("Perceptual Loss (VGG Feature Distance)")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(run_dir / cfg["Loss_Plot_name"])

wandb.finish()

# Testing

visualize_batch(trained_enc, trained_dec, val_loader)



