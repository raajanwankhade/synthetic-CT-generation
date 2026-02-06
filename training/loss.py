import torch
import torch.nn as nn

from config import cfg
from training.metrics import ssim_loss
from scripts.segmentor import SemanticConsistencyModule


LOSS_NAMES = {
    "total": "total_loss",
    "l1": "l1_loss",
    "ssim": "ssim_loss",
    "semantic": "semantic_loss",
}

_LOSS_CTX = None


def _get_loss_context(device: torch.device):
    global _LOSS_CTX
    if _LOSS_CTX is None or _LOSS_CTX.get("device") != device:
        _LOSS_CTX = {
            "device": device,
            "criterion": nn.L1Loss(),
            "semantic_criterion": SemanticConsistencyModule(device),
        }
    return _LOSS_CTX


def compute_total_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
):
    ctx = _get_loss_context(preds.device)
    loss_l1 = ctx["criterion"](preds, target)
    loss_ssim = ssim_loss(preds, target, int(cfg.perf.ssim_downsample))
    loss_sem = ctx["semantic_criterion"](preds, mask)
    loss = loss_l1 + cfg.training.lambda_ssim * loss_ssim + cfg.training.lambda_sem * loss_sem
    return {
        LOSS_NAMES["total"]: loss,
        LOSS_NAMES["l1"]: loss_l1,
        LOSS_NAMES["ssim"]: loss_ssim,
        LOSS_NAMES["semantic"]: loss_sem,
    }
