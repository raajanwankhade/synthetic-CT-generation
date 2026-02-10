import torch
import torch.nn.functional as F
import numpy as np


class ImageMetricsGPU:
    def __init__(self, debug=False, device="cuda"):
        self.dynamic_range = (-1024., 3071.)
        self.debug = debug
        self.device = torch.device(device)

    # ---------- helpers ----------

    def _to_tensor(self, arr):
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    def _mask_tensor(self, mask):
        if mask is None:
            return None
        return (self._to_tensor(mask) > 0).float()

    # ---------- MAE ----------

    def mae(self, gt, pred, mask=None):
        gt = self._to_tensor(gt)
        pred = self._to_tensor(pred)
        mask = self._mask_tensor(mask)

        if mask is None:
            return torch.mean(torch.abs(gt - pred)).item()

        diff = torch.abs(gt - pred) * mask
        return (diff.sum() / mask.sum()).item()

    # ---------- PSNR ----------

    def psnr(self, gt, pred, mask=None, use_population_range=False):
        gt = self._to_tensor(gt)
        pred = self._to_tensor(pred)
        mask = self._mask_tensor(mask)

        if use_population_range:
            lo, hi = self.dynamic_range
            gt = gt.clamp(lo, hi)
            pred = pred.clamp(lo, hi)
            dr = hi - lo
        else:
            lo, hi = gt.min(), gt.max()
            pred = pred.clamp(lo, hi)
            dr = hi - lo

        if mask is not None:
            gt = gt[mask == 1]
            pred = pred[mask == 1]

        mse = torch.mean((gt - pred) ** 2)
        return (20 * torch.log10(dr) - 10 * torch.log10(mse)).item()

    # ---------- uniform filter (3D) ----------

    def _uniform_filter3d(self, x, k=7):
        kernel = torch.ones((1, 1, k, k, k), device=self.device) / (k ** 3)
        x = x.unsqueeze(0).unsqueeze(0)
        out = F.conv3d(x, kernel, padding=k//2)
        return out.squeeze()

    # ---------- SSIM scale ----------

    def structural_similarity_at_scale(
        self, im1, im2,
        luminance_weight,
        contrast_weight,
        structure_weight,
        data_range,
        win_size=7
    ):
        ux = self._uniform_filter3d(im1, win_size)
        uy = self._uniform_filter3d(im2, win_size)

        uxx = self._uniform_filter3d(im1 * im1, win_size)
        uyy = self._uniform_filter3d(im2 * im2, win_size)
        uxy = self._uniform_filter3d(im1 * im2, win_size)

        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy

        R = data_range
        K1, K2 = 0.01, 0.03
        C1 = (K1 * R) ** 2
        C2 = (K2 * R) ** 2
        C3 = C2 / 2

        L = (2 * ux * uy + C1) / (ux * ux + uy * uy + C1)
        C = (2 * torch.sqrt(vx.clamp_min(0)) * torch.sqrt(vy.clamp_min(0)) + C2) / (vx + vy + C2)
        S = (vxy + C3) / (torch.sqrt(vx.clamp_min(0)) * torch.sqrt(vy.clamp_min(0)) + C3)

        ssim_map = (L ** luminance_weight) * (C ** contrast_weight) * (S ** structure_weight)

        pad = win_size // 2
        cropped = ssim_map[pad:-pad, pad:-pad, pad:-pad]

        return cropped.mean(), cropped

    # ---------- MS-SSIM ----------

    def ms_ssim(self, gt, pred, mask=None):
        gt = self._to_tensor(gt)
        pred = self._to_tensor(pred)
        mask = self._mask_tensor(mask)

        lo, hi = self.dynamic_range
        gt = gt.clamp(lo, hi) - lo
        pred = pred.clamp(lo, hi) - lo

        if mask is not None:
            gt *= mask
            pred *= mask

        dr = hi - lo

        weights = torch.tensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
            device=self.device
        )

        levels = len(weights)

        ms_vals = []
        ms_mask_vals = []

        for i in range(levels):
            ssim_val, ssim_map = self.structural_similarity_at_scale(
                gt, pred,
                luminance_weight=0 if i < levels - 1 else weights[i],
                contrast_weight=weights[i],
                structure_weight=weights[i],
                data_range=dr
            )

            ms_vals.append(ssim_val)

            if mask is not None:
                m = mask[3:-3, 3:-3, 3:-3]
                ms_mask_vals.append(ssim_map[m.bool()].mean())

            gt = F.avg_pool3d(gt.unsqueeze(0).unsqueeze(0), 2).squeeze()
            pred = F.avg_pool3d(pred.unsqueeze(0).unsqueeze(0), 2).squeeze()

            if mask is not None:
                mask = mask[::2, ::2, ::2]

        ms = torch.prod(torch.stack(ms_vals))
        ms_mask = torch.prod(torch.stack(ms_mask_vals)) if mask is not None else ms

        return ms.item(), ms_mask.item()

    # ---------- main API ----------

    def score_patient(self, gt, pred, mask):
        mae = self.mae(gt, pred, mask)
        psnr = self.psnr(gt, pred, mask, True)
        _, ms = self.ms_ssim(gt, pred, mask)

        return {
            "mae": mae,
            "psnr": psnr,
            "ms_ssim": ms
        }
