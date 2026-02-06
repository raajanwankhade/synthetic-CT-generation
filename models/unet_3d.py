import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    """3D residual block: Conv3d -> IN -> LeakyReLU (x N) + skip."""

    def __init__(
        self,
        channels: int,
        convs_per_block: int = 2,
        norm_layer=nn.InstanceNorm3d,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        if convs_per_block < 1:
            raise ValueError("convs_per_block must be >= 1")

        layers = []
        for _ in range(convs_per_block):
            layers.extend(
                [
                    nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
                    norm_layer(channels, affine=True),
                    nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class ResEncoder3D(nn.Module):
    """3D residual encoder that returns one feature per downsampling level."""

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        num_stages: int = 4,
        blocks_per_stage: int | list[int] = 2,
        stage_channels: list[int] | None = None,
        convs_per_block: int = 2,
        norm_layer=nn.InstanceNorm3d,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, kernel_size=7, stride=1, padding=3, bias=False),
            norm_layer(base_ch, affine=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

        stages = []
        if stage_channels is None:
            stage_channels = [base_ch * (2 ** i) for i in range(num_stages)]
        if isinstance(blocks_per_stage, int):
            blocks_per_stage = [blocks_per_stage for _ in range(len(stage_channels))]
        if len(stage_channels) != len(blocks_per_stage):
            raise ValueError("stage_channels and blocks_per_stage must have the same length")

        ch = stage_channels[0]
        for s, (stage_ch, n_blocks) in enumerate(zip(stage_channels, blocks_per_stage)):
            if stage_ch != ch:
                stages.append(
                    nn.Sequential(
                        nn.Conv3d(ch, stage_ch, kernel_size=1, bias=False),
                        norm_layer(stage_ch, affine=True),
                        nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                    )
                )
                ch = stage_ch

            blocks = [
                ResBlock3D(
                    ch,
                    convs_per_block=convs_per_block,
                    norm_layer=norm_layer,
                    negative_slope=negative_slope,
                )
                for _ in range(n_blocks)
            ]
            stage = nn.Sequential(*blocks)
            stages.append(stage)
            if s != len(stage_channels) - 1:
                next_ch = stage_channels[s + 1]
                down = nn.Sequential(
                    nn.Conv3d(ch, next_ch, kernel_size=3, stride=2, padding=1, bias=False),
                    norm_layer(next_ch, affine=True),
                    nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                )
                stages.append(down)
                ch = next_ch

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


class UpConvBlock3D(nn.Module):
    """3D Conv + IN + LeakyReLU + optional residual blocks (upsampling outside)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        blocks: int = 1,
        convs_per_block: int = 2,
        norm_layer=nn.InstanceNorm3d,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm_layer(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        ]
        for _ in range(blocks):
            layers.append(
                ResBlock3D(
                    out_ch,
                    convs_per_block=convs_per_block,
                    norm_layer=norm_layer,
                    negative_slope=negative_slope,
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResDecoder3D(nn.Module):
    """3D decoder that reconstructs an image with the same spatial size as feats[0]."""

    def __init__(
        self,
        encoder_feature_channels: list[int],
        out_ch: int = 1,
        blocks_per_up: int = 1,
        convs_per_block: int = 2,
        norm_layer=nn.InstanceNorm3d,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.feat_chs = encoder_feature_channels
        self.up_blocks = nn.ModuleList()

        for i in range(len(encoder_feature_channels) - 1, 0, -1):
            in_ch = encoder_feature_channels[i]
            skip_ch = encoder_feature_channels[i - 1]
            self.up_blocks.append(
                UpConvBlock3D(
                    in_ch + skip_ch,
                    skip_ch,
                    blocks=blocks_per_up,
                    convs_per_block=convs_per_block,
                    norm_layer=norm_layer,
                    negative_slope=negative_slope,
                )
            )

        final_in_ch = encoder_feature_channels[0]
        self.final_conv = nn.Sequential(
            nn.Conv3d(final_in_ch, final_in_ch // 2, kernel_size=3, padding=1, bias=False),
            norm_layer(final_in_ch // 2, affine=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv3d(final_in_ch // 2, out_ch, kernel_size=1),
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        x = feats[-1]
        for i, up in enumerate(self.up_blocks):
            skip_idx = len(feats) - 2 - i
            skip = feats[skip_idx]
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = up(x)
        return self.final_conv(x)


def build_resunet3d(cfg) -> tuple[ResEncoder3D, ResDecoder3D]:
    """Factory for a 3D ResUNet using Model3DConfig or dict."""
    # Support both Pydantic config and dict
    if hasattr(cfg, "stage_channels"):
        stage_channels = cfg.stage_channels
    else:
        stage_channels = cfg.get("stage_channels")
    
    if hasattr(cfg, "base_ch"):
        base_ch = cfg.base_ch
        in_ch = cfg.in_ch
        out_ch = cfg.out_ch
        blocks_per_stage = cfg.blocks_per_stage
        convs_per_block = cfg.convs_per_block
        negative_slope = cfg.negative_slope
        num_stages = getattr(cfg, "num_stages", 4)
    else:
        base_ch = cfg.get("base_ch", 32)
        in_ch = cfg["in_ch"]
        out_ch = cfg["out_ch"]
        blocks_per_stage = cfg["blocks_per_stage"]
        convs_per_block = cfg["convs_per_block"]
        negative_slope = cfg["negative_slope"]
        num_stages = cfg.get("num_stages", 4)
    
    if stage_channels is None:
        stage_channels = [base_ch * (2 ** i) for i in range(num_stages)]
    
    enc = ResEncoder3D(
        in_ch=in_ch,
        base_ch=stage_channels[0] if stage_channels else base_ch,
        num_stages=len(stage_channels),
        blocks_per_stage=blocks_per_stage,
        stage_channels=stage_channels,
        convs_per_block=convs_per_block,
        norm_layer=nn.InstanceNorm3d,
        negative_slope=negative_slope,
    )
    return enc, ResDecoder3D(
        encoder_feature_channels=stage_channels,
        out_ch=out_ch,
        blocks_per_up=1,
        convs_per_block=convs_per_block,
        norm_layer=nn.InstanceNorm3d,
        negative_slope=negative_slope,
    )
