from __future__ import annotations

import torch.nn as nn

from config import cfg
from .base import GenerativeModel, UNetWrapper
from .unet_2d import build_resunet2d
from .unet_3d import build_resunet3d


def build_model(is_3d: bool) -> GenerativeModel:
    """
    Build a GenerativeModel wrapper.
    
    Returns a unified model with forward(), state_dict_components(), etc.
    To add new architectures (GAN, Diffusion), add branches here.
    """
    if is_3d:
        enc, dec = build_resunet3d(cfg.model3d)
    else:
        enc, dec = build_resunet2d(
            in_ch=1,
            out_ch=1,
            base_ch=32,
            num_stages=4,
            blocks_per_stage=2,
        )
    return UNetWrapper(enc, dec)
