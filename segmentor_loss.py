
import importlib.util
import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from monai.losses.dice import DiceLoss
from monai.networks.nets.unet import UNet


DEFAULT_CONFIG_PATH = "config.py"


def _load_cfg(config_path: Optional[str] = None):
    """Load `cfg` from a Python config module path."""
    path = config_path or os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)
    spec = importlib.util.spec_from_file_location("config", path)
    print(f"[segmentor_loss.py]Loading config from {path} ...")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cfg


cfg = _load_cfg()

class SegmentorLoss(nn.Module):
    """
    Frozen-teacher semantic consistency loss.

    Loads the teacher segmentor trained by ``train_teacher_segmentor.py``,
    freezes it, and computes Dice loss between the teacher's prediction on
    the generated sCT and the ground-truth mask.

    Normalization notes
    -------------------
    * The teacher was trained on **population z-scored CT**
      (clip [-1024, 3000] → z = (ct - pop_mean) / pop_std).
    * The generator's sCT output is trained to match that same CT space.
    * We load the population stats so we can clamp the sCT to the valid
      z-score range seen during teacher training, protecting against wild
      generator outputs early in training.
    * The mask must be binarised to match teacher training
      (``(mask > 0.5).float()``).
    """

    # Fallback z-score clamp when population stats are unavailable.
    _DEFAULT_ZSCORE_MIN = -5.0
    _DEFAULT_ZSCORE_MAX = 10.0

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        # -- Population CT stats for z-score clamping -------------------
        self._z_min, self._z_max = self._load_ct_zscore_bounds()

        # -- Build teacher with the same architecture used during training
        teacher_cfg = cfg.get("teacher_cfg", {})
        stage_channels = tuple(teacher_cfg.get("stage_channels", [32, 64, 128, 256]))
        strides = tuple([2] * (len(stage_channels) - 1))
        is_3d = str(cfg.get("num_adjacent_slices", 0)).lower() == "3d"

        self.teacher = UNet(
            spatial_dims=3 if is_3d else 2,
            in_channels=int(teacher_cfg.get("in_ch", 1)),
            out_channels=int(teacher_cfg.get("out_ch", 1)),
            channels=stage_channels,
            strides=strides,
            num_res_units=int(teacher_cfg.get("num_res_units", 2)),
            norm=teacher_cfg.get("norm", "instance"),
        ).to(device)

        # -- Load trained weights
        weights_path = (
            Path(cfg["run_dir"])
            / cfg["ckpt_dir"]
            / cfg.get("teacher_filename", "frozen_teacher.pth")
        )
        print(f"Loading Teacher weights from {weights_path} ...")
        ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        self.teacher.load_state_dict(state_dict)

        # Freeze teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Dice loss (sigmoid=True because teacher outputs raw logits)
        self.dice_loss_fn = DiceLoss(sigmoid=True, include_background=False)

    # ------------------------------------------------------------------ #
    #  helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_ct_zscore_bounds():
        """Derive the valid z-score range from population CT stats.

        The preprocessing pipeline clips CT to [CLIP_MIN, CLIP_MAX], then
        z-scores: ``z = (ct - mean) / std``.
        We replicate that to get the z-score bounds.
        """
        CLIP_MIN, CLIP_MAX = -1024, 3000
        stats_dir = Path(cfg["root"]) / "stats"
        mean_path = stats_dir / "mean.npy"
        std_path = stats_dir / "std.npy"

        if mean_path.exists() and std_path.exists():
            pop_mean = float(np.load(str(mean_path)))
            pop_std = float(np.load(str(std_path)))
            if pop_std < 1e-6:
                pop_std = 1e-6
            z_min = (CLIP_MIN - pop_mean) / pop_std
            z_max = (CLIP_MAX - pop_mean) / pop_std
            print(
                f"[SegmentorLoss] CT population stats loaded: "
                f"mean={pop_mean:.2f}, std={pop_std:.2f} → "
                f"z-range=[{z_min:.2f}, {z_max:.2f}]"
            )
            return float(z_min), float(z_max)

        print(
            "[SegmentorLoss] WARNING: population stats not found at "
            f"{stats_dir}; using default z-score clamp "
            f"[{SegmentorLoss._DEFAULT_ZSCORE_MIN}, "
            f"{SegmentorLoss._DEFAULT_ZSCORE_MAX}]"
        )
        return SegmentorLoss._DEFAULT_ZSCORE_MIN, SegmentorLoss._DEFAULT_ZSCORE_MAX

    # Keep teacher in eval mode even if parent module is set to train
    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, sCT_pred: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sCT_pred:  (B, 1, D, H, W) or (B, C, H, W) — synthetic CT from
                       the generator (population z-scored CT space).
            gt_mask:   same shape — raw organ mask (will be binarised here).
        Returns:
            Scalar Dice loss (0 when no foreground present).
        """
        # 1. Binarise mask to match teacher training (mask > 0.5)
        gt_mask = (gt_mask > 0.5).float()

        # 2. Check for foreground — works for both 4-D (2D) and 5-D (3D)
        spatial_dims = tuple(range(1, gt_mask.dim()))  # (1,2,3) or (1,2,3,4)
        has_organ = gt_mask.sum(dim=spatial_dims) > 10
        if has_organ.sum() == 0:
            return sCT_pred.sum() * 0.0  # zero with grad attachment

        # 3. Clamp sCT to the z-score range the teacher was trained on.
        #    Uses a soft clamp (hardtanh) so gradients still flow at boundaries.
        sCT_clamped = torch.nn.functional.hardtanh(
            sCT_pred, min_val=self._z_min, max_val=self._z_max
        )

        # 4. Teacher forward — gradients flow back to sCT_pred only
        #    (teacher params are frozen, requires_grad=False)
        teacher_logits = self.teacher(sCT_clamped)

        loss = self.dice_loss_fn(
            teacher_logits[has_organ],
            gt_mask[has_organ],
        )
        return loss
