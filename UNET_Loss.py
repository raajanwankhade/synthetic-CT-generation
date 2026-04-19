import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr

class LossModule(nn.Module):
    def __init__(self, device, ssim_down=1, ssim_weight=1.0, seg_weight=0.1, return_items=True):
        super().__init__()
        self.ssim_down = ssim_down
        self.ssim_weight = ssim_weight
        self.seg_weight = float(seg_weight)
        self.return_items = bool(return_items)
        self.device = device

        self.l1_loss = nn.L1Loss()
        self.segmentor_loss = None
        if self.seg_weight > 0.0:
            # Import lazily so segmentor_loss reads CONFIG_PATH after runtime setup.
            from segmentor_loss import SegmentorLoss
            self.segmentor_loss = SegmentorLoss(device)

    @staticmethod
    def _as_tensor(metric_out: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if isinstance(metric_out, tuple):
            return metric_out[0]
        return metric_out
    
    def _ssim_loss(self, preds: torch.Tensor, y: torch.Tensor, ssim_down: int) -> torch.Tensor:
        if ssim_down > 1:
            if preds.dim() == 5:
                preds = F.avg_pool3d(preds, kernel_size=ssim_down)
                y = F.avg_pool3d(y, kernel_size=ssim_down)
            else:
                preds = F.avg_pool2d(preds, kernel_size=ssim_down)
                y = F.avg_pool2d(y, kernel_size=ssim_down)
        # Compute data_range from target (z-score normalized, unbounded)
        data_range = (y.max() - y.min()).clamp(min=1e-8)
        ssim_score = self._as_tensor(ssim(preds, y, data_range=data_range))
        # Clamp to valid SSIM range [-1, 1] for numerical safety only;
        # preserves full gradient signal from negative SSIM (severe mismatch)
        ssim_score = torch.clamp(ssim_score, -1.0, 1.0)
        return 1 - ssim_score
    
    
    @torch.no_grad()
    def compute_val_metrics(self, preds: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
        data_range = (target.max() - target.min()).clamp(min=1e-8)

        ssim_val = self._as_tensor(ssim(preds, target, data_range=data_range)).item()
        psnr_val = self._as_tensor(psnr(preds, target, data_range=data_range)).item()
        return ssim_val, psnr_val
    
    def forward(self, preds: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        l1 = self.l1_loss(preds, y)
        ssim_loss = self._ssim_loss(preds, y, self.ssim_down)
        if self.segmentor_loss is not None:
            seg_loss = self.segmentor_loss(preds, mask)
        else:
            seg_loss = preds.new_zeros(())
        total_loss = l1 + self.ssim_weight * ssim_loss + self.seg_weight * seg_loss
        if self.return_items:
            return total_loss, l1.item(), ssim_loss.item(), seg_loss.item()
        return total_loss, l1, ssim_loss, seg_loss

    # def surrogate_dice_loss(
    #     self,
    #     logits: torch.Tensor,
    #     pseudo_labels: torch.Tensor,
    #     eps: float = 1e-6,
    # ) -> torch.Tensor:
    #     probs = torch.sigmoid(logits)
    #     b, c = probs.shape[:2]

    #     probs_flat = probs.view(b, c, -1)
    #     labels_flat = pseudo_labels.view(b, c, -1).to(probs_flat.dtype)

    #     intersection = (probs_flat * labels_flat).sum(dim=-1)
    #     cardinality = probs_flat.sum(dim=-1) + labels_flat.sum(dim=-1)
    #     dice_per_class = (2.0 * intersection + eps) / (cardinality + eps)
    #     return 1.0 - dice_per_class.mean()


