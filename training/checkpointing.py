import os
import torch

from models.base import GenerativeModel


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

    def update(self, epoch, model: GenerativeModel, optimizer, scheduler, scaler, val_metrics):
        if self.monitor not in val_metrics:
            raise KeyError(f"Monitor key '{self.monitor}' not in val_metrics: {list(val_metrics.keys())}")
        score = float(val_metrics[self.monitor])
        improved = (score < self.best) if self.mode == "min" else (score > self.best)
        if improved:
            self.best = score
            self.best_epoch = int(epoch)
            self.best_metrics = dict(val_metrics)
            ckpt_path = os.path.join(self.save_dir, self.filename)
            self.save_epoch(ckpt_path, epoch, model, optimizer, scheduler, scaler)
            self.best_path = ckpt_path
            print(f"✅ Saved BEST checkpoint: epoch={epoch} {self.monitor}={score:.5f} -> {ckpt_path}")

    def save_epoch(self, ckpt_path, epoch, model: GenerativeModel, optimizer, scheduler, scaler):
        ckpt = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }
        # Save model components (encoder/decoder for UNet, generator/discriminator for GAN, etc.)
        ckpt.update(model.state_dict_components())
        torch.save(ckpt, ckpt_path)

    def load_best(self, model: GenerativeModel, optimizer=None, device=None):
        if not self.best_path:
            candidate = os.path.join(self.save_dir, self.filename)
            if os.path.exists(candidate):
                self.best_path = candidate
            else:
                raise ValueError("No best checkpoint found.")
        ckpt = torch.load(self.best_path, map_location=device, weights_only=False)
        model.load_state_dict_components(ckpt)
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        epoch = ckpt.get("epoch", None)
        print(f"🔁 Loaded best checkpoint from {self.best_path} (epoch={epoch})")
        return ckpt
