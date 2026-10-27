import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateSegHead(nn.Module):
    def __init__(self, n_classes: int = 1) -> None:
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.head = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, n_classes, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, sct_pred: torch.Tensor) -> torch.Tensor:
        x = self.pool(sct_pred)
        logits_small = self.head(x)
        logits = F.interpolate(
            logits_small,
            size=sct_pred.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        return logits