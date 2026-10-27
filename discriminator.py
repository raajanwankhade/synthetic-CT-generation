"""UNet3D-inspired configurable discriminator for volumetric data.

This module defines a 3D CNN discriminator that follows the residual,
multi-stage downsampling pattern used by the UNet3D encoder.

Important:
- The final layer returns raw logits.
- Do not apply sigmoid in this model; use BCEWithLogitsLoss externally.
- Architecture can be configured from cfg["discriminator3d_cfg"] in config.py.
"""

import importlib.util
import os
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn


DEFAULT_CONFIG_PATH = "config.py"


def _load_cfg(config_path: Optional[str] = None):
    """Load cfg from config.py or from CONFIG_PATH when set."""
    path = config_path or os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)
    spec = importlib.util.spec_from_file_location("config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cfg


def _conv_norm_lrelu3d(
    in_ch: int,
    out_ch: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    norm_layer=nn.InstanceNorm3d,
    negative_slope: float = 0.2,
) -> nn.Sequential:
    """3D Conv -> InstanceNorm -> LeakyReLU helper."""
    return nn.Sequential(
        nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        norm_layer(out_ch, affine=True),
        nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
    )


class ResBlock3D(nn.Module):
    """3D residual block: Conv3d -> IN -> LeakyReLU (x N) + skip."""

    def __init__(
        self,
        channels: int,
        convs_per_block: int = 2,
        norm_layer=nn.InstanceNorm3d,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        if convs_per_block < 1:
            raise ValueError("convs_per_block must be >= 1")

        layers: List[nn.Module] = []
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


class PlainBlock3D(nn.Module):
    """3D non-residual block: Conv3d -> IN -> LeakyReLU (x N)."""

    def __init__(
        self,
        channels: int,
        convs_per_block: int = 2,
        norm_layer=nn.InstanceNorm3d,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        if convs_per_block < 1:
            raise ValueError("convs_per_block must be >= 1")

        layers: List[nn.Module] = []
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
        return self.net(x)


def _resolve_norm_layer3d(norm: Union[str, type[nn.Module]]) -> type[nn.Module]:
    """Resolve a norm config value to a 3D normalization layer class."""
    if isinstance(norm, str):
        key = norm.lower().strip()
        if key in ("instance", "instancenorm", "instancenorm3d", "in"):
            return nn.InstanceNorm3d
        if key in ("batch", "batchnorm", "batchnorm3d", "bn"):
            return nn.BatchNorm3d
        raise ValueError(f"Unsupported norm '{norm}'. Use 'instance' or 'batch'.")

    if issubclass(norm, nn.Module):
        return norm
    raise TypeError("norm must be a string or a nn.Module class")


class Discriminator3D(nn.Module):
    """Configurable 3D discriminator inspired by the UNet3D encoder.

    By default, this returns patch-level logits with shape [B, 1, D', H', W'].
    If pool_logits=True, it returns image-level logits with shape [B, 1].
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        num_stages: int = 4,
        blocks_per_stage: Union[int, Sequence[int]] = 2,
        stage_channels: Optional[Sequence[int]] = None,
        convs_per_block: int = 2,
        norm_layer=nn.InstanceNorm3d,
        negative_slope: float = 0.2,
        pool_logits: bool = False,
        use_residual: bool = True,
    ):
        super().__init__()
        if stage_channels is None:
            stage_channels = [base_ch * (2 ** i) for i in range(num_stages)]
        if len(stage_channels) == 0:
            raise ValueError("stage_channels must contain at least one stage")

        if isinstance(blocks_per_stage, int):
            blocks_per_stage = [blocks_per_stage for _ in range(len(stage_channels))]
        if len(stage_channels) != len(blocks_per_stage):
            raise ValueError("stage_channels and blocks_per_stage must have the same length")

        first_ch = int(stage_channels[0])
        self.stem = _conv_norm_lrelu3d(
            in_ch,
            first_ch,
            kernel_size=7,
            stride=1,
            padding=3,
            norm_layer=norm_layer,
            negative_slope=negative_slope,
        )

        stages: List[nn.Module] = []
        ch = first_ch
        block_cls = ResBlock3D if use_residual else PlainBlock3D
        for s, (stage_ch, n_blocks) in enumerate(zip(stage_channels, blocks_per_stage)):
            stage_ch = int(stage_ch)
            if stage_ch != ch:
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
                block_cls(
                    ch,
                    convs_per_block=convs_per_block,
                    norm_layer=norm_layer,
                    negative_slope=negative_slope,
                )
                for _ in range(n_blocks)
            ]
            stages.append(nn.Sequential(*blocks))

            if s != len(stage_channels) - 1:
                next_ch = int(stage_channels[s + 1])
                stages.append(
                    _conv_norm_lrelu3d(
                        ch,
                        next_ch,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        norm_layer=norm_layer,
                        negative_slope=negative_slope,
                    )
                )
                ch = next_ch

        self.stages = nn.ModuleList(stages)
        self.logits_head = nn.Sequential(
            _conv_norm_lrelu3d(
                ch,
                ch,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=norm_layer,
                negative_slope=negative_slope,
            ),
            nn.Conv3d(ch, 1, kernel_size=1, bias=True),
        )
        self.pool_logits = pool_logits
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for m in self.stages:
            x = m(x)

        logits = self.logits_head(x)
        if self.pool_logits:
            return logits.mean(dim=(2, 3, 4))
        return logits


def build_discriminator3d(disc_cfg: dict) -> Discriminator3D:
    """Factory helper to build a Discriminator3D from a config dictionary."""
    stage_channels = disc_cfg.get("stage_channels")
    if stage_channels is None:
        num_stages = disc_cfg.get("num_stages", 4)
        base_ch = disc_cfg.get("base_ch", 32)
        stage_channels = [base_ch * (2 ** i) for i in range(num_stages)]

    norm_layer = _resolve_norm_layer3d(disc_cfg.get("norm", "instance"))
    use_residual = disc_cfg.get("use_residual", disc_cfg.get("residual", True))

    return Discriminator3D(
        in_ch=disc_cfg.get("in_ch", 1),
        base_ch=disc_cfg.get("base_ch", int(stage_channels[0])),
        num_stages=len(stage_channels),
        blocks_per_stage=disc_cfg.get("blocks_per_stage", 2),
        stage_channels=stage_channels,
        convs_per_block=disc_cfg.get("convs_per_block", 2),
        norm_layer=norm_layer,
        negative_slope=disc_cfg.get("negative_slope", 0.2),
        pool_logits=disc_cfg.get("pool_logits", False),
        use_residual=use_residual,
    )


def build_discriminator3d_from_project_cfg(
    project_cfg: dict,
    cfg_key: str = "discriminator3d_cfg",
) -> Discriminator3D:
    """Build Discriminator3D from a dedicated config dictionary in cfg."""
    if cfg_key not in project_cfg:
        raise KeyError(
            f"'{cfg_key}' not found in cfg. Add cfg['{cfg_key}'] to config.py."
        )
    return build_discriminator3d(project_cfg[cfg_key])


def build_discriminator3d_from_config(
    config_path: Optional[str] = None,
    cfg_key: str = "discriminator3d_cfg",
) -> Discriminator3D:
    """Load cfg from config.py (or CONFIG_PATH) and build the discriminator."""
    project_cfg = _load_cfg(config_path=config_path)
    return build_discriminator3d_from_project_cfg(project_cfg=project_cfg, cfg_key=cfg_key)


if __name__ == "__main__":
    model = build_discriminator3d_from_config(cfg_key="discriminator3d_cfg")
    x = torch.randn(2, 1, 32, 192, 192)
    logits = model(x)
    print("Input shape:", tuple(x.shape))
    print("Logits shape (no sigmoid):", tuple(logits.shape))
    print("Residual enabled:", model.use_residual)