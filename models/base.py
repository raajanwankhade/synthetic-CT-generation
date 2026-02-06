"""Base class for all generative models."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class GenerativeModel(nn.Module):
    """
    Unified wrapper for generative models (UNet, GAN, Diffusion, etc.).
    
    Subclasses must implement:
      - forward(x) -> output
      - state_dict_components() -> Dict[str, state_dict]
      - load_state_dict_components(state)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def state_dict_components(self) -> Dict[str, dict]:
        """Return named state dicts for checkpointing."""
        raise NotImplementedError

    def load_state_dict_components(self, state: Dict[str, dict]):
        """Load named state dicts from checkpoint."""
        raise NotImplementedError


class UNetWrapper(GenerativeModel):
    """Wraps encoder + decoder into a single GenerativeModel."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.decoder(feats)

    def state_dict_components(self) -> Dict[str, dict]:
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
        }

    def load_state_dict_components(self, state: Dict[str, dict]):
        if "encoder" in state:
            self.encoder.load_state_dict(state["encoder"])
        if "decoder" in state:
            self.decoder.load_state_dict(state["decoder"])
