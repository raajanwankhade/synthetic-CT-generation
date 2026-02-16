import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr

class LossModule(nn.Module):
    def __init__(self, device, ssim_down=1, ssim_weight=1.0):
        super(LossModule, self).__init__()
        self.ssim_down = ssim_down
        self.ssim_weight = ssim_weight
        self.device = device

        self.l1_loss = nn.L1Loss()
    
    def _ssim_loss(self, preds_hu: torch.Tensor, y_norm: torch.Tensor, ssim_down: int) -> torch.Tensor:
        preds_norm = torch.clamp(preds_hu, 0, 1)
        # y_norm = torch.clamp((y_norm + 1024) / 4095, 0, 1)
        if ssim_down > 1:
            if preds_norm.dim() == 5:
                preds_norm = F.avg_pool3d(preds_norm, kernel_size=ssim_down)
                y_norm = F.avg_pool3d(y_norm, kernel_size=ssim_down)
            else:
                preds_norm = F.avg_pool2d(preds_norm, kernel_size=ssim_down)
                y_norm = F.avg_pool2d(y_norm, kernel_size=ssim_down)
        # torchmetrics SSIM can return values in [-1, 1] or exceed bounds due to numerical issues
        # Clamp to [0, 1] to ensure loss is in valid [0, 1] range
        ssim_score = torch.clamp(ssim(preds_norm, y_norm, data_range=1.0), 0.0, 1.0)
        return 1 - ssim_score 

    @torch.no_grad()
    def compute_val_metrics(self, pred_norm, target_norm):
        pred_norm = torch.clamp(pred_norm, 0, 1)
        target_norm = torch.clamp(target_norm, 0, 1)

        ssim_val = ssim(pred_norm, target_norm, data_range=1.0).item()
        psnr_val = psnr(pred_norm, target_norm, data_range=1.0).item()
        return ssim_val, psnr_val
    
    def forward(self, preds: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        l1 = self.l1_loss(preds, y)
        ssim_loss = self._ssim_loss(preds, y, self.ssim_down)
        total_loss = l1 + self.ssim_weight * ssim_loss
        return total_loss, l1.item(), ssim_loss.item()


