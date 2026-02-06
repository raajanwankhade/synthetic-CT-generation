import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr


@torch.no_grad()
def compute_metrics(pred, target, ct_range=(-1024, 3071)):
    min_hu, max_hu = ct_range
    data_range = max_hu - min_hu

    pred_norm = (pred - min_hu) / data_range
    target_norm = (target - min_hu) / data_range

    pred_norm = torch.clamp(pred_norm, 0, 1)
    target_norm = torch.clamp(target_norm, 0, 1)

    ssim_val = ssim(pred_norm, target_norm, data_range=1.0).item()
    psnr_val = psnr(pred_norm, target_norm, data_range=1.0).item()
    return ssim_val, psnr_val


def ssim_loss(preds_hu: torch.Tensor, y_hu: torch.Tensor, ssim_down: int) -> torch.Tensor:
    preds_norm = torch.clamp((preds_hu + 1024) / 4095, 0, 1)
    y_norm = torch.clamp((y_hu + 1024) / 4095, 0, 1)
    if ssim_down > 1:
        if preds_norm.dim() == 5:
            preds_norm = F.avg_pool3d(preds_norm, kernel_size=ssim_down)
            y_norm = F.avg_pool3d(y_norm, kernel_size=ssim_down)
        else:
            preds_norm = F.avg_pool2d(preds_norm, kernel_size=ssim_down)
            y_norm = F.avg_pool2d(y_norm, kernel_size=ssim_down)
    ssim_val = torch.clamp(ssim(preds_norm, y_norm, data_range=1.0), 0.0, 1.0)
    return 1 - ssim_val
