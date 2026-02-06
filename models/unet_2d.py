import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Simple residual block: conv -> norm -> ReLU -> conv -> norm + skip"""

    def __init__(self, channels: int, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = norm_layer(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_layer(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ResEncoder2D(nn.Module):
    """Residual encoder that returns one feature per downsampling level."""

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        num_stages: int = 4,
        blocks_per_stage: int = 2,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=7, stride=1, padding=3, bias=False),
            norm_layer(base_ch),
            nn.ReLU(inplace=True),
        )

        stages = []
        ch = base_ch
        for s in range(num_stages):
            blocks = [ResBlock(ch, norm_layer=norm_layer) for _ in range(blocks_per_stage)]
            stage = nn.Sequential(*blocks)
            stages.append(stage)
            if s != num_stages - 1:
                down = nn.Sequential(
                    nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    norm_layer(ch * 2),
                    nn.ReLU(inplace=True),
                )
                stages.append(down)
                ch *= 2

        self.stages = nn.ModuleList(stages)
        self.out_ch = ch

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        x = self.stem(x)
        for m in self.stages:
            x = m(x)
            if len(feats) == 0 or feats[-1].shape[-1] != x.shape[-1]:
                feats.append(x)
        return feats


class UpConvBlock(nn.Module):
    """Conv + norm + ReLU + optional residual blocks (upsampling is done outside)."""

    def __init__(self, in_ch: int, out_ch: int, blocks: int = 1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks):
            layers.append(ResBlock(out_ch, norm_layer=norm_layer))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResDecoder2D(nn.Module):
    """Decoder that reconstructs an image with the same spatial size as feats[0]."""

    def __init__(
        self,
        encoder_feature_channels: list[int],
        out_ch: int = 1,
        blocks_per_up: int = 1,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        self.feat_chs = encoder_feature_channels
        self.up_blocks = nn.ModuleList()

        for i in range(len(encoder_feature_channels) - 1, 0, -1):
            in_ch = encoder_feature_channels[i]
            skip_ch = encoder_feature_channels[i - 1]
            self.up_blocks.append(
                UpConvBlock(in_ch + skip_ch, skip_ch, blocks=blocks_per_up, norm_layer=norm_layer)
            )

        final_in_ch = encoder_feature_channels[0]
        self.final_conv = nn.Sequential(
            nn.Conv2d(final_in_ch, final_in_ch // 2, kernel_size=3, padding=1, bias=False),
            norm_layer(final_in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_in_ch // 2, out_ch, kernel_size=1),
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        x = feats[-1]
        for i, up in enumerate(self.up_blocks):
            skip_idx = len(feats) - 2 - i
            skip = feats[skip_idx]
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = up(x)
        out = self.final_conv(x)
        return out


def build_resunet2d(
    in_ch: int = 1,
    out_ch: int = 1,
    base_ch: int = 32,
    num_stages: int = 4,
    blocks_per_stage: int = 2,
    stage_channels: list[int] | None = None,
    norm_layer=nn.BatchNorm2d,
) -> tuple[ResEncoder2D, ResDecoder2D]:
    """Factory for a 2D ResUNet."""
    if stage_channels is None:
        stage_channels = [base_ch * (2 ** i) for i in range(num_stages)]
    enc = ResEncoder2D(
        in_ch=in_ch,
        base_ch=stage_channels[0],
        num_stages=len(stage_channels),
        blocks_per_stage=blocks_per_stage,
        norm_layer=norm_layer,
    )
    dec = ResDecoder2D(encoder_feature_channels=stage_channels, out_ch=out_ch, norm_layer=norm_layer)
    return enc, dec
