"""2D/3D residual U-Net style encoder/decoder building blocks.

This module contains:
- 2D residual encoder/decoder blocks
- 3D residual encoder/decoder blocks
- factory helpers for constructing complete 2D/3D models
- lightweight local test helpers
"""

import importlib.util
import os
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = "config.py"


def _load_cfg(config_path: Optional[str] = None):
    """Load `cfg` from a Python config module path."""
    path = config_path or os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)
    spec = importlib.util.spec_from_file_location("config", path)
    print(f"[unet_model.py]Loading config from {path} ...")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cfg


cfg = _load_cfg()


# -----------------------------------------------------------------------------
# Core residual blocks and shared helpers
# -----------------------------------------------------------------------------

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


class ResBlock2D(nn.Module):
    """2D residual block: Conv2d -> IN -> LeakyReLU (x N) + skip."""

    def __init__(
        self,
        channels: int,
        convs_per_block: int = 2,
        norm_layer=nn.InstanceNorm2d,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        if convs_per_block < 1:
            raise ValueError("convs_per_block must be >= 1")

        layers = []
        for _ in range(convs_per_block):
            layers.extend(
                [
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    norm_layer(channels, affine=True),
                    nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


def _conv_norm_lrelu3d(
    in_ch: int,
    out_ch: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    norm_layer=nn.InstanceNorm3d,
    negative_slope: float = 0.01,
) -> nn.Sequential:
    """3D Conv -> InstanceNorm -> LeakyReLU helper."""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        norm_layer(out_ch, affine=True),
        nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
    )


def _conv_norm_relu2d(
    in_ch: int,
    out_ch: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    norm_layer=nn.BatchNorm2d,
) -> nn.Sequential:
    """2D Conv -> BatchNorm -> ReLU helper."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        norm_layer(out_ch),
        nn.ReLU(inplace=True),
    )


def _conv_norm_lrelu2d(
    in_ch: int,
    out_ch: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    norm_layer=nn.InstanceNorm2d,
    negative_slope: float = 0.01,
) -> nn.Sequential:
    """2D Conv -> InstanceNorm -> LeakyReLU helper."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        norm_layer(out_ch, affine=True),
        nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
    )


def _infer_stage_channels(model_cfg: dict) -> List[int]:
    """Return explicit encoder stage channels from a model config."""
    stage_channels = model_cfg.get("stage_channels")
    if stage_channels is not None:
        return stage_channels
    num_stages = model_cfg.get("num_stages", 4)
    base_ch = model_cfg["base_ch"]
    return [base_ch * (2 ** i) for i in range(num_stages)]


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
        self.stem = _conv_norm_lrelu3d(
            in_ch,
            base_ch,
            kernel_size=7,
            stride=1,
            padding=3,
            norm_layer=norm_layer,
            negative_slope=negative_slope,
        )

        stages = []
        self.stage_positions = []
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
                    _conv_norm_lrelu3d(
                        ch,
                        stage_ch,
                        kernel_size=1,
                        norm_layer=norm_layer,
                        negative_slope=negative_slope,
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
            self.stage_positions.append(len(stages) - 1)
            if s != len(stage_channels) - 1:
                next_ch = stage_channels[s + 1]
                down = _conv_norm_lrelu3d(
                    ch,
                    next_ch,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_layer=norm_layer,
                    negative_slope=negative_slope,
                )
                stages.append(down)
                ch = next_ch

        self.stages = nn.ModuleList(stages)
        self.out_ch = ch

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = torch.jit.annotate(List[torch.Tensor], [])
        x = self.stem(x)
        for idx, m in enumerate(self.stages):
            x = m(x)
            if idx in self.stage_positions:
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
        layers: List[nn.Module] = [
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
        encoder_feature_channels: Sequence[int],
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
            _conv_norm_lrelu3d(
                final_in_ch,
                final_in_ch // 2,
                kernel_size=3,
                padding=1,
                norm_layer=norm_layer,
                negative_slope=negative_slope,
            ),
            nn.Conv3d(final_in_ch // 2, out_ch, kernel_size=1),
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        x = feats[-1]
        for i, up in enumerate(self.up_blocks):
            skip_idx = len(feats) - 2 - i
            skip = feats[skip_idx]
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = up(x)
        return self.final_conv(x)


class ResEncoder2D(nn.Module):
    """2D residual encoder that returns one feature per downsampling level."""

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        num_stages: int = 4,
        blocks_per_stage: Union[int, List[int]] = 2,
        stage_channels: Optional[List[int]] = None,
        convs_per_block: int = 2,
        norm_layer=nn.InstanceNorm2d,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.stem = _conv_norm_lrelu2d(
            in_ch,
            base_ch,
            kernel_size=7,
            stride=1,
            padding=3,
            norm_layer=norm_layer,
            negative_slope=negative_slope,
        )

        stages = []
        self.stage_positions = []
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
                    _conv_norm_lrelu2d(
                        ch,
                        stage_ch,
                        kernel_size=1,
                        norm_layer=norm_layer,
                        negative_slope=negative_slope,
                    )
                )
                ch = stage_ch

            blocks = [
                ResBlock2D(
                    ch,
                    convs_per_block=convs_per_block,
                    norm_layer=norm_layer,
                    negative_slope=negative_slope,
                )
                for _ in range(n_blocks)
            ]
            stage = nn.Sequential(*blocks)
            stages.append(stage)
            self.stage_positions.append(len(stages) - 1)
            if s != len(stage_channels) - 1:
                next_ch = stage_channels[s + 1]
                down = _conv_norm_lrelu2d(
                    ch,
                    next_ch,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_layer=norm_layer,
                    negative_slope=negative_slope,
                )
                stages.append(down)
                ch = next_ch

        self.stages = nn.ModuleList(stages)
        self.out_ch = ch

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = torch.jit.annotate(List[torch.Tensor], [])
        x = self.stem(x)
        for idx, m in enumerate(self.stages):
            x = m(x)
            if idx in self.stage_positions:
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


class UpConvBlock2D(nn.Module):
    """2D Conv + IN + LeakyReLU + optional residual blocks (upsampling outside)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        blocks: int = 1,
        convs_per_block: int = 2,
        norm_layer=nn.InstanceNorm2d,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm_layer(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        ]
        for _ in range(blocks):
            layers.append(
                ResBlock2D(
                    out_ch,
                    convs_per_block=convs_per_block,
                    norm_layer=norm_layer,
                    negative_slope=negative_slope,
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResDecoder2D(nn.Module):
    """2D decoder that reconstructs an image with the same spatial size as feats[0]."""

    def __init__(
        self,
        encoder_feature_channels: Sequence[int],
        out_ch: int = 1,
        blocks_per_up: int = 1,
        convs_per_block: int = 2,
        norm_layer=nn.InstanceNorm2d,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.feat_chs = encoder_feature_channels
        self.up_blocks = nn.ModuleList()

        for i in range(len(encoder_feature_channels) - 1, 0, -1):
            in_ch = encoder_feature_channels[i]
            skip_ch = encoder_feature_channels[i - 1]
            self.up_blocks.append(
                UpConvBlock2D(
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
            _conv_norm_lrelu2d(
                final_in_ch,
                final_in_ch // 2,
                kernel_size=3,
                padding=1,
                norm_layer=norm_layer,
                negative_slope=negative_slope,
            ),
            nn.Conv2d(final_in_ch // 2, out_ch, kernel_size=1),
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        x = feats[-1]
        for i, up in enumerate(self.up_blocks):
            skip_idx = len(feats) - 2 - i
            skip = feats[skip_idx]
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = up(x)
        return self.final_conv(x)


# -----------------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------------

def build_resunet3d(model_cfg: dict) -> tuple[ResEncoder3D, ResDecoder3D]:
    """Factory for a 3D ResUNet; returns `(enc, dec)`."""
    stage_channels = _infer_stage_channels(model_cfg)
    enc = ResEncoder3D(
        in_ch=model_cfg["in_ch"],
        base_ch=model_cfg.get("base_ch", stage_channels[0]),
        num_stages=len(stage_channels),
        blocks_per_stage=model_cfg["blocks_per_stage"],
        stage_channels=stage_channels,
        convs_per_block=model_cfg["convs_per_block"],
        norm_layer=nn.InstanceNorm3d,
        negative_slope=model_cfg["negative_slope"],
    )
    dec = ResDecoder3D(
        encoder_feature_channels=stage_channels,
        out_ch=model_cfg["out_ch"],
        blocks_per_up=model_cfg.get("blocks_per_up", 1),
        convs_per_block=model_cfg["convs_per_block"],
        norm_layer=nn.InstanceNorm3d,
        negative_slope=model_cfg["negative_slope"],
    )
    return enc, dec


def build_resunet2d(model_cfg: dict) -> tuple[ResEncoder2D, ResDecoder2D]:
    """Factory for a 2D ResUNet; returns `(enc, dec)`."""
    stage_channels = _infer_stage_channels(model_cfg)
    enc = ResEncoder2D(
        in_ch=model_cfg["in_ch"],
        base_ch=model_cfg.get("base_ch", stage_channels[0]),
        num_stages=len(stage_channels),
        blocks_per_stage=model_cfg["blocks_per_stage"],
        stage_channels=stage_channels,
        convs_per_block=model_cfg["convs_per_block"],
        norm_layer=nn.InstanceNorm2d,
        negative_slope=model_cfg["negative_slope"],
    )
    dec = ResDecoder2D(
        encoder_feature_channels=stage_channels,
        out_ch=model_cfg["out_ch"],
        blocks_per_up=model_cfg.get("blocks_per_up", 1),
        convs_per_block=model_cfg["convs_per_block"],
        norm_layer=nn.InstanceNorm2d,
        negative_slope=model_cfg["negative_slope"],
    )
    return enc, dec


# -----------------------------------------------------------------------------
# Testing helpers (moved from UNET_Train.py)
# -----------------------------------------------------------------------------

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_encoder(input_size=(4, 1, 256, 256)):
    device = _get_device()
    enc = ResEncoder2D(in_ch=1, base_ch=32, num_stages=4, blocks_per_stage=2).to(device)
    x = torch.randn(*input_size, device=device)
    feats = enc(x)
    print("Number of returned feature maps:", len(feats))
    for i, f in enumerate(feats):
        print(f"feat[{i}] shape: {tuple(f.shape)}")
    return feats


def test_encoder_decoder(input_size=(4, 1, 256, 256)):
    device = _get_device()
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
    device = _get_device()
    enc, dec = build_resunet3d(cfg["model3d_cfg"])
    print("Encoder:", enc)
    print("Decoder:", dec)
    enc = enc.to(device)
    dec = dec.to(device)
    dummy = torch.randn(*input_size, device=device)
    feats = enc(dummy)
    feat_chs = [f.shape[1] for f in feats]
    print("Encoder feature channels (3D):", feat_chs)
    for i, f in enumerate(feats):
        print(f"feat[{i}] shape:", f.shape)
    out = dec(enc(torch.randn(*input_size, device=device)))
    print("Output shape (3D):", out.shape)
    return out

def test_encoder_decoder_2d(input_size=(2, 1, 256, 256)):
    device = _get_device()
    enc, dec = build_resunet2d(cfg["model2d_cfg"])
    print("Encoder:", enc)
    print("Decoder:", dec)
    enc = enc.to(device)
    dec = dec.to(device)
    dummy = torch.randn(*input_size, device=device)
    feats = enc(dummy)
    feat_chs = [f.shape[1] for f in feats]
    print("Encoder feature channels (2D):", feat_chs)
    for i, f in enumerate(feats):
        print(f"feat[{i}] shape:", f.shape)
    out = dec(enc(torch.randn(*input_size, device=device)))
    print("Output shape (2D):", out.shape)
    return out

if __name__ == "__main__":
    print("Testing ResEncoder2D + ResDecoder2D...")
    test_encoder_decoder_2d(input_size=(2, 1, 256, 256))
    # print("\nTesting ResEncoder3D + ResDecoder3D...")
    # test_encoder_decoder_3d(input_size=(2, 1, 40, 128, 128))
    # =========================================================================
    # torch.compile testing — DISABLED
    # Triton 3.1.0 (required by PyTorch 2.5.1) cannot handle CUDA 12.8+ driver.
    # Re-enable after upgrading to PyTorch 2.6+ with compatible Triton.
    # =========================================================================
    print("\n⚠️  torch.compile tests skipped (Triton 3.1.0 / CUDA 12.9 incompatibility)")