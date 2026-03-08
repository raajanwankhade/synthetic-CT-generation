"""Export a trained ResUNet (2D or 3D) to a self-contained TorchScript archive.

The exported .pt file contains:
  - The full fused encoder+decoder graph (torch.jit.script, not trace)
  - Embedded model_cfg JSON so the architecture is always recoverable

Loading at inference time is just:
    model = torch.jit.load("model.pt", map_location="cpu")
    out   = model(input_tensor)

No need to import UNET_model, redefine the architecture, or call load_state_dict.

Optionally exports to ONNX as well (--onnx).
"""

import argparse
import json
import os
import importlib.util
from io import BytesIO

import torch
import torch.nn as nn
from typing import List

DEFAULT_CONFIG_PATH = "config.py"
config_path = os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)

spec = importlib.util.spec_from_file_location("config", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
cfg = config_module.cfg

from UNET_model import build_resunet2d, build_resunet3d


# =========================
# CLI ARGUMENTS
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Export trained ResUNet to self-contained TorchScript (+ optional ONNX)"
    )
    parser.add_argument("--model_type", type=str, required=True, choices=["2d", "3d"],
                        help="Model type: '2d' or '3d'")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file (.pth)")
    parser.add_argument("--export_path", type=str, required=True,
                        help="Path to save TorchScript model (.pt)")
    parser.add_argument("--onnx", action="store_true",
                        help="Also export an ONNX file (same path with .onnx extension)")
    # Input shapes (used for verification & ONNX export)
    parser.add_argument("--input_shape_2d", type=int, nargs=4, default=[1, 1, 192, 192],
                        help="Input shape for 2D model: B C H W (default: 1 1 192 192)")
    parser.add_argument("--input_shape_3d", type=int, nargs=5, default=[1, 1, 32, 192, 192],
                        help="Input shape for 3D model: B C D H W (default: 1 1 32 192 192)")
    return parser.parse_args()


args = parse_args()

# =========================
# SETTINGS
# =========================
MODEL_TYPE = args.model_type
CHECKPOINT_PATH = args.checkpoint
EXPORT_PATH = args.export_path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if MODEL_TYPE == "2d":
    INPUT_SHAPE = tuple(args.input_shape_2d)
    model_cfg = cfg["model2d_cfg"]
else:
    INPUT_SHAPE = tuple(args.input_shape_3d)
    model_cfg = cfg["model3d_cfg"]

print(f"Model type : {MODEL_TYPE}")
print(f"Input shape: {INPUT_SHAPE}")
print(f"Device     : {DEVICE}")
print(f"Model cfg  : {json.dumps(model_cfg, indent=2)}")

# =========================
# LOAD CHECKPOINT
# =========================
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

# =========================
# BUILD MODEL VIA FACTORY
# =========================
if MODEL_TYPE == "2d":
    enc, dec = build_resunet2d(model_cfg)
else:
    enc, dec = build_resunet3d(model_cfg)

enc.load_state_dict(ckpt["encoder"])
dec.load_state_dict(ckpt["decoder"])


# =========================
# FUSED MODEL (scriptable)
# =========================
class FullModel(torch.nn.Module):
    """Single-module wrapper so the export is one callable graph."""
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: List[torch.Tensor] = self.encoder(x)
        return self.decoder(feats)


model = FullModel(enc, dec).to(DEVICE)
model.eval()

# =========================
# SCRIPT (not trace)
# =========================
# torch.jit.script captures control-flow (e.g. if idx in stage_positions)
# faithfully, unlike trace which bakes in a single execution path.
print("Scripting model …")
scripted_model = torch.jit.script(model)

# Embed model_cfg + model_type as extra metadata inside the archive.
# Recoverable later via: torch.jit.load(…)._extra_files
metadata = {
    "model_type": MODEL_TYPE,
    "model_cfg": model_cfg,
    "input_shape": list(INPUT_SHAPE),
}
extra_files = {"model_metadata.json": json.dumps(metadata, indent=2)}

# =========================
# SAVE
# =========================
os.makedirs(os.path.dirname(os.path.abspath(EXPORT_PATH)), exist_ok=True)
torch.jit.save(scripted_model, EXPORT_PATH, _extra_files=extra_files)
print(f"Saved TorchScript model to: {EXPORT_PATH}")

# =========================
# VERIFY NUMERICAL MATCH
# =========================
load_extra = {"model_metadata.json": ""}
loaded_model = torch.jit.load(EXPORT_PATH, map_location=DEVICE, _extra_files=load_extra)
loaded_meta = json.loads(load_extra["model_metadata.json"])
print(f"Embedded metadata: {json.dumps(loaded_meta, indent=2)}")

with torch.no_grad():
    x_test = torch.randn(*INPUT_SHAPE, device=DEVICE)
    y_original = model(x_test)
    y_loaded = loaded_model(x_test)
    max_diff = torch.max(torch.abs(y_original - y_loaded)).item()

print(f"Max absolute difference: {max_diff}")
if max_diff < 1e-5:
    print("✔ TorchScript export verified successfully.")
else:
    print("⚠ WARNING: Output mismatch. Do NOT trust this export.")

# =========================
# OPTIONAL ONNX EXPORT
# =========================
if args.onnx:
    onnx_path = os.path.splitext(EXPORT_PATH)[0] + ".onnx"
    print(f"Exporting ONNX to: {onnx_path}")
    with torch.no_grad():
        dummy = torch.randn(*INPUT_SHAPE, device=DEVICE)
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
    print(f"✔ ONNX saved to: {onnx_path}")

# =========================
# USAGE REMINDER
# =========================
print("\n─── Inference usage (no architecture code needed) ───")
print(f'  model = torch.jit.load("{EXPORT_PATH}", map_location="cpu")')
print(f'  output = model(torch.randn{INPUT_SHAPE})')
print(f'\n  # To inspect embedded config:')
print(f'  extra = {{"model_metadata.json": ""}}')
print(f'  model = torch.jit.load("{EXPORT_PATH}", _extra_files=extra)')
print(f'  import json; cfg = json.loads(extra["model_metadata.json"])')