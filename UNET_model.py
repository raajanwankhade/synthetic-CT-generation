import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
from config import cfg
# -----------------------------------------------------------------------------
# Configurable model settings
# -----------------------------------------------------------------------------
# You can edit these values to build 3D models consistently from a single object.

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


# -----------------------------------------------------------------------------
# 3D variants (InstanceNorm3d + LeakyReLU)
# -----------------------------------------------------------------------------

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
        blocks_per_stage: Union[int, List[int]] = 2,
        stage_channels: Optional[List[int]] = None,
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
                # adjust channels before blocks if stage_ch differs
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

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = torch.jit.annotate(List[torch.Tensor], [])
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

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        x = feats[-1]
        for i, up in enumerate(self.up_blocks):
            skip_idx = len(feats) - 2 - i
            skip = feats[skip_idx]
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = up(x)
        return self.final_conv(x)


def build_resunet3d(cfg: dict) -> tuple[ResEncoder3D, ResDecoder3D]:
    """Factory for a 3D ResUNet using MODEL_CFG-like dict."""
    stage_channels = cfg.get("stage_channels")
    if stage_channels is None:
        stage_channels = [cfg["base_ch"] * (2 ** i) for i in range(cfg.get("num_stages", 4))]
    enc = ResEncoder3D(
        in_ch=cfg["in_ch"],
        base_ch=cfg.get("base_ch", stage_channels[0]),
        num_stages=len(stage_channels),
        blocks_per_stage=cfg["blocks_per_stage"],
        stage_channels=stage_channels,
        convs_per_block=cfg["convs_per_block"],
        norm_layer=nn.InstanceNorm3d,
        negative_slope=cfg["negative_slope"],
    )
    return enc, ResDecoder3D(
        encoder_feature_channels=stage_channels,
        out_ch=cfg["out_ch"],
        blocks_per_up=1,
        convs_per_block=cfg["convs_per_block"],
        norm_layer=nn.InstanceNorm3d,
        negative_slope=cfg["negative_slope"],
    )


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

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = torch.jit.annotate(List[torch.Tensor], [])
        x = self.stem(x)
        for m in self.stages:
            x = m(x)
            # only save feature maps that change spatial resolution (avoid duplicate same-size)
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

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        x = feats[-1]  # start from deepest (lowest res)
        for i, up in enumerate(self.up_blocks):
            skip_idx = len(feats) - 2 - i
            skip = feats[skip_idx]
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = up(x)
        out = self.final_conv(x)
        return out


# -----------------------------------------------------------------------------
# Testing helpers (moved from UNET_Train.py)
# -----------------------------------------------------------------------------

def test_encoder(input_size=(4, 1, 256, 256)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = ResEncoder2D(in_ch=1, base_ch=32, num_stages=4, blocks_per_stage=2).to(device)
    x = torch.randn(*input_size, device=device)
    feats = enc(x)
    print("Number of returned feature maps:", len(feats))
    for i, f in enumerate(feats):
        print(f"feat[{i}] shape: {tuple(f.shape)}")
    return feats


def test_encoder_decoder(input_size=(4, 1, 256, 256)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = ResEncoder2D(in_ch=1, base_ch=32, num_stages=4, blocks_per_stage=2).to(device)
    dummy = torch.randn(*input_size, device=device)
    feats = enc(dummy)
    feat_chs = [f.shape[1] for f in feats]
    print("Encoder feature channels:", feat_chs)
    for i, f in enumerate(feats):
        print(f"feat[{i}] shape:", f.shape)

    dec = ResDecoder2D(encoder_feature_channels=feat_chs, out_ch=1).to(device)
    out = dec(enc(torch.randn(*input_size, device=device)))
    print("Output shape:", out.shape)
    return out


def test_encoder_decoder_3d(input_size=(2, 1, 40, 128, 128)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = ResEncoder3D(
        in_ch=cfg["model3d_cfg"]["in_ch"],
        base_ch=cfg["model3d_cfg"]["base_ch"],
        num_stages=len(cfg["model3d_cfg"].get("stage_channels", [])) or 4,
        blocks_per_stage=cfg["model3d_cfg"]["blocks_per_stage"],
        stage_channels=cfg["model3d_cfg"].get("stage_channels"),
        convs_per_block=cfg["model3d_cfg"]["convs_per_block"],
        norm_layer=nn.InstanceNorm3d,
        negative_slope=cfg["model3d_cfg"]["negative_slope"],
    ).to(device)
    dummy = torch.randn(*input_size, device=device)
    feats = enc(dummy)
    feat_chs = [f.shape[1] for f in feats]
    print("Encoder feature channels (3D):", feat_chs)
    for i, f in enumerate(feats):
        print(f"feat[{i}] shape:", f.shape)

    dec = ResDecoder3D(
        encoder_feature_channels=feat_chs,
        out_ch=cfg["model3d_cfg"]["out_ch"],
        convs_per_block=cfg["model3d_cfg"]["convs_per_block"],
        norm_layer=nn.InstanceNorm3d,
        negative_slope=cfg["model3d_cfg"]["negative_slope"],
    ).to(device)
    out = dec(enc(torch.randn(*input_size, device=device)))
    print("Output shape (3D):", out.shape)
    return out


def build_resunet2d(
    in_ch: int = 1,
    out_ch: int = 1,
    base_ch: int = 32,
    num_stages: int = 4,
    blocks_per_stage: int = 2,
    stage_channels: Optional[List[int]] = None,
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

if __name__ == "__main__":
    print("Testing ResEncoder2D...")
    test_encoder(input_size=(2, 1, 64, 64))
    print("\nTesting ResEncoder2D + ResDecoder2D...")
    test_encoder_decoder(input_size=(2, 1, 64, 64))
    print("\nTesting ResEncoder3D + ResDecoder3D...")
    test_encoder_decoder_3d(input_size=(2, 1, 40, 128, 128))