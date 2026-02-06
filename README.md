# Synthetic CT Generation

A modular PyTorch framework for MR-to-CT / CBCT-to-CT image translation using deep learning. Designed to be **model-agnostic** — supports UNet, GAN, Diffusion, or any custom architecture with minimal changes.

---

## Project Structure

```
bhuvanesh/
├── config/                 # Configuration management
│   ├── config.yaml         # Main config file
│   ├── schemas.py          # Pydantic validation schemas
│   └── loader.py           # Config loader
├── data/                   # Data loading & preprocessing
│   ├── dataset.py          # MRCTDataset class
│   ├── loaders.py          # DataLoader builders
│   ├── patches.py          # Patch extraction utilities
│   └── normalization.py    # Intensity normalization
├── models/                 # Model architectures
│   ├── base.py             # GenerativeModel base class
│   ├── factory.py          # Model factory (build_model)
│   ├── unet_2d.py          # 2D ResUNet
│   └── unet_3d.py          # 3D ResUNet
├── training/               # Training utilities
│   ├── trainer.py          # Main training loop
│   ├── loss.py             # Loss computation
│   ├── metrics.py          # SSIM, PSNR metrics
│   ├── optim.py            # Optimizer & scheduler
│   ├── checkpointing.py    # Checkpoint management
│   ├── callbacks.py        # Training callbacks
│   ├── visualization.py    # Plotting utilities
│   └── wandb_utils.py      # Weights & Biases logging
├── inference/              # Inference & evaluation
│   ├── model_test.py       # Inference pipeline
│   └── calc_metrics.py     # Metric calculation
├── scripts/                # Entry point scripts
│   ├── UNET_Train.py       # Main training script
│   ├── preproc.py          # Preprocessing script
│   └── segmentor.py        # Semantic consistency module
├── slurm/                  # SLURM job scripts
└── utils/                  # Shared utilities
    ├── constants.py        # Global constants
    └── env.py              # Environment setup
```

---

## Quick Start

### 1. Install Dependencies

```bash
conda activate sctgen
conda install --file requirements.txt
```

### 2. Configure Training

Edit `config/config.yaml`:

```yaml
paths:
  run_dir: ./runs/my_experiment
  root: /path/to/dataset

data:
  modality: mr              # 'mr' or 'cbct'
  anatomy: HN               # 'HN', 'AB', or 'TH'
  num_adjacent_slices: "3d" # '3d' for volumetric, 0/1/3/5 for 2D/2.5D

training:
  epochs: 100
  lr: 0.001
  lambda_ssim: 100.0
  lambda_sem: 0.1
```

### 3. Run Training

```bash
python scripts/UNET_Train.py
```

---

## Model Architecture

### GenerativeModel Base Class

All models inherit from `GenerativeModel` in `models/base.py`:

```python
class GenerativeModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate output from input."""
        raise NotImplementedError

    def state_dict_components(self) -> Dict[str, dict]:
        """Return named state dicts for checkpointing."""
        raise NotImplementedError

    def load_state_dict_components(self, state: Dict[str, dict]):
        """Load named state dicts from checkpoint."""
        raise NotImplementedError
```

### Current UNet Implementation

```python
class UNetWrapper(GenerativeModel):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def state_dict_components(self):
        return {"encoder": self.encoder.state_dict(), 
                "decoder": self.decoder.state_dict()}
```

---

## Adding a New Model

### Example 1: Adding a GAN

1. **Create `models/gan.py`:**

```python
import torch
import torch.nn as nn
from models.base import GenerativeModel


class Generator(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()
        # Your generator architecture (e.g., Pix2Pix, CycleGAN)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 7, padding=3),
            nn.ReLU(inplace=True),
            # ... more layers
            nn.Conv2d(base_ch, out_ch, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, in_ch=2, base_ch=64):  # in_ch=2 for conditional (input+output)
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # ... PatchGAN discriminator
            nn.Conv2d(base_ch * 8, 1, 4, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class GANWrapper(GenerativeModel):
    """Wraps Generator + Discriminator for GAN training."""

    def __init__(self, generator: nn.Module, discriminator: nn.Module):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate output (inference mode uses generator only)."""
        return self.generator(x)

    def state_dict_components(self):
        return {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }

    def load_state_dict_components(self, state):
        if "generator" in state:
            self.generator.load_state_dict(state["generator"])
        if "discriminator" in state:
            self.discriminator.load_state_dict(state["discriminator"])


def build_gan(cfg) -> GANWrapper:
    gen = Generator(in_ch=cfg.in_ch, out_ch=cfg.out_ch, base_ch=cfg.base_ch)
    disc = Discriminator(in_ch=cfg.in_ch + cfg.out_ch, base_ch=cfg.base_ch)
    return GANWrapper(gen, disc)
```

2. **Update `models/factory.py`:**

```python
from models.base import GenerativeModel, UNetWrapper
from models.unet_2d import build_resunet2d
from models.unet_3d import build_resunet3d
from models.gan import build_gan  # Add import


def build_model(model_type: str, cfg) -> GenerativeModel:
    if model_type == "unet_3d":
        enc, dec = build_resunet3d(cfg.model3d)
        return UNetWrapper(enc, dec)
    elif model_type == "unet_2d":
        enc, dec = build_resunet2d(in_ch=1, out_ch=1, base_ch=32)
        return UNetWrapper(enc, dec)
    elif model_type == "gan":
        return build_gan(cfg.gan)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
```

3. **Add GAN config to `config/schemas.py`:**

```python
class GANConfig(BaseModel):
    in_ch: int = 1
    out_ch: int = 1
    base_ch: int = 64
    adversarial_weight: float = 1.0
    l1_weight: float = 100.0
```

4. **Create a GAN-specific training script** `scripts/GAN_Train.py`:

GAN training requires alternating discriminator/generator updates, so it needs a separate training script:

```python
# scripts/GAN_Train.py
from models.gan import build_gan
from training.optim import build_optimizer, build_scheduler
from training.checkpointing import CheckpointManager

model = build_gan(cfg.gan)
model.to(device)

# Separate optimizers for G and D
opt_g = build_optimizer(model.generator.parameters(), device, cfg.training.lr_g)
opt_d = build_optimizer(model.discriminator.parameters(), device, cfg.training.lr_d)

adv_criterion = nn.BCEWithLogitsLoss()
l1_criterion = nn.L1Loss()

for epoch in range(num_epochs):
    for batch in train_loader:
        x = batch["input"].to(device)
        y = batch["label"].to(device)
        
        # ---------------------
        # Train Discriminator
        # ---------------------
        opt_d.zero_grad()
        
        # Real
        real_input = torch.cat([x, y], dim=1)
        pred_real = model.discriminator(real_input)
        loss_d_real = adv_criterion(pred_real, torch.ones_like(pred_real))
        
        # Fake (detach to avoid training G)
        fake = model.generator(x).detach()
        fake_input = torch.cat([x, fake], dim=1)
        pred_fake = model.discriminator(fake_input)
        loss_d_fake = adv_criterion(pred_fake, torch.zeros_like(pred_fake))
        
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        opt_d.step()
        
        # ---------------------
        # Train Generator
        # ---------------------
        opt_g.zero_grad()
        
        fake = model.generator(x)
        fake_input = torch.cat([x, fake], dim=1)
        pred_fake = model.discriminator(fake_input)
        
        # Adversarial loss (fool discriminator)
        loss_g_adv = adv_criterion(pred_fake, torch.ones_like(pred_fake))
        # L1 reconstruction loss
        loss_g_l1 = l1_criterion(fake, y)
        
        loss_g = loss_g_adv + cfg.training.lambda_l1 * loss_g_l1
        loss_g.backward()
        opt_g.step()
    
    # Checkpointing still works - saves both G and D
    ckpt_mgr.update(epoch, model, opt_g, scheduler, scaler, val_metrics)
```

> **Note:** GAN training requires a separate script because of the alternating G/D training pattern. The `GANWrapper` still provides unified checkpointing and inference (`model(x)` uses generator only).

---

### Example 2: Adding a Diffusion Model

1. **Create `models/diffusion.py`:**

```python
import torch
import torch.nn as nn
from models.base import GenerativeModel


class DenoisingUNet(nn.Module):
    """U-Net for noise prediction in diffusion models."""
    def __init__(self, in_ch=2, out_ch=1, base_ch=64, time_emb_dim=256):
        super().__init__()
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        # Your denoising U-Net architecture
        # ...

    def forward(self, x, t):
        """
        Args:
            x: noisy image (B, C, H, W)
            t: timestep (B,)
        Returns:
            predicted noise
        """
        t_emb = self.time_mlp(t.unsqueeze(-1).float())
        # ... forward pass with time conditioning
        return predicted_noise


class DiffusionWrapper(GenerativeModel):
    """Wraps denoising model + noise scheduler for diffusion."""

    def __init__(self, denoiser: nn.Module, num_timesteps: int = 1000):
        super().__init__()
        self.denoiser = denoiser
        self.num_timesteps = num_timesteps
        # Beta schedule
        self.register_buffer("betas", torch.linspace(1e-4, 0.02, num_timesteps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: run full denoising loop."""
        device = x.device
        # Start from pure noise
        img = torch.randn_like(x)
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            # Concatenate conditioning (input MR) with noisy image
            model_input = torch.cat([x, img], dim=1)
            pred_noise = self.denoiser(model_input, t_batch)
            # DDPM sampling step
            img = self._denoise_step(img, pred_noise, t)
        return img

    def _denoise_step(self, x, pred_noise, t):
        # Standard DDPM denoising step
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        beta = self.betas[t]
        
        mean = (1 / alpha.sqrt()) * (x - (beta / (1 - alpha_cumprod).sqrt()) * pred_noise)
        if t > 0:
            noise = torch.randn_like(x)
            return mean + beta.sqrt() * noise
        return mean

    def training_loss(self, x_0, condition):
        """Compute diffusion training loss."""
        t = torch.randint(0, self.num_timesteps, (x_0.size(0),), device=x_0.device)
        noise = torch.randn_like(x_0)
        # q(x_t | x_0)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = alpha_cumprod_t.sqrt() * x_0 + (1 - alpha_cumprod_t).sqrt() * noise
        # Predict noise
        model_input = torch.cat([condition, x_t], dim=1)
        pred_noise = self.denoiser(model_input, t)
        return nn.functional.mse_loss(pred_noise, noise)

    def state_dict_components(self):
        return {"denoiser": self.denoiser.state_dict()}

    def load_state_dict_components(self, state):
        if "denoiser" in state:
            self.denoiser.load_state_dict(state["denoiser"])
```

2. **For diffusion, you may need a custom training protocol** since the loss computation differs. Create `training/protocols/diffusion.py` or modify `training/loss.py` to handle diffusion loss.

---

## Checkpointing

Checkpoints are model-agnostic. The `CheckpointManager` uses:

```python
model.state_dict_components()  # Save
model.load_state_dict_components(ckpt)  # Load
```

This means checkpoints automatically save the right components:
- **UNet**: `{"encoder": ..., "decoder": ...}`
- **GAN**: `{"generator": ..., "discriminator": ...}`
- **Diffusion**: `{"denoiser": ...}`

---

## Loss Functions

Current loss in `training/loss.py`:

```python
loss = L1 + λ_ssim * SSIM_loss + λ_sem * Semantic_loss
```

To add custom losses for GAN/Diffusion:

```python
# In training/loss.py
def compute_gan_loss(gen_out, target, disc_real, disc_fake, cfg):
    l1_loss = nn.L1Loss()(gen_out, target)
    adv_loss = nn.BCEWithLogitsLoss()(disc_fake, torch.ones_like(disc_fake))
    return {
        "total_loss": cfg.l1_weight * l1_loss + cfg.adv_weight * adv_loss,
        "l1_loss": l1_loss,
        "adv_loss": adv_loss,
    }
```

---

## Configuration Reference

### Key Config Sections

| Section | Description |
|---------|-------------|
| `paths` | Run directory, dataset root |
| `data` | Modality, anatomy, patch size, slices |
| `dataloader` | Batch size, workers, prefetch |
| `perf` | AMP dtype, channels_last |
| `training` | Epochs, LR, loss weights |
| `checkpoint` | Checkpoint directory, filenames |
| `model3d` | 3D UNet config |
| `logging` | WandB settings |

---

## Inference

```bash
python inference/model_test.py
```

Or use programmatically:

```python
from models import build_model
import torch

model = build_model(is_3d=True)
ckpt = torch.load("checkpoints/best.pth")
model.load_state_dict_components(ckpt)
model.eval()

output = model(input_tensor)
```

---

## Tips

1. **Switching models**: Change factory call in training script
2. **Custom losses**: Extend `training/loss.py` with new loss functions
3. **New architectures**: Implement `GenerativeModel` interface
4. **Checkpoints are compatible**: Same checkpoint manager works for all models