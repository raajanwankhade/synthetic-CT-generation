from typing import Iterable

import torch
import torch.nn as nn


def get_amp_dtype(cfg) -> torch.dtype:
    amp_dtype_name = str(cfg.perf.amp_dtype).lower()
    return torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16


def build_optimizer(params: Iterable[nn.Parameter], device: torch.device, lr: float):
    """
    Build optimizer from an iterable of parameters.
    
    Use model.parameters() for a unified model.
    """
    params_list = list(params)
    try:
        optimizer = torch.optim.AdamW(
            params_list,
            lr=lr,
            fused=(device.type == "cuda"),
        )
        if device.type == "cuda":
            print("✅ Using AdamW(fused=True) on CUDA")
        return optimizer
    except TypeError:
        return torch.optim.Adam(
            params_list,
            weight_decay=0,
            lr=lr,
        )


def build_scheduler(optimizer, scheduler_patience: int):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience
    )
