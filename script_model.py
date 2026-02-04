import argparse
import torch
import torch.nn as nn

from config import cfg
from UNET_model import ResEncoder2D, ResDecoder2D, ResEncoder3D, ResDecoder3D
# =========================
# CLI ARGUMENTS
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Export trained model to TorchScript and upload to Kaggle")
    parser.add_argument("--model_type", type=str, required=True, choices=["2d", "3d"],
                        help="Model type: '2d' or '3d'")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file (.pth)")
    parser.add_argument("--export_path", type=str, required=True,
                        help="Path to save TorchScript model")
    # Input shapes (override defaults)
    parser.add_argument("--input_shape_2d", type=int, nargs=4, default=[1, 1, 256, 256],
                        help="Input shape for 2D model: B C H W (default: 1 1 256 256)")
    parser.add_argument("--input_shape_3d", type=int, nargs=5, default=[1, 1, 32, 192, 160],
                        help="Input shape for 3D model: B C D H W (default: 1 1 32 192 160)")
    return parser.parse_args()

args = parse_args()

# =========================
# SETTINGS FROM ARGS + CONFIG
# =========================
MODEL_TYPE = args.model_type
CHECKPOINT_PATH = args.checkpoint
EXPORT_PATH = args.export_path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if MODEL_TYPE == "2d":
    INPUT_SHAPE = tuple(args.input_shape_2d)
else:
    INPUT_SHAPE = tuple(args.input_shape_3d)

print(f"Model type: {MODEL_TYPE}")
print(f"Input shape: {INPUT_SHAPE}")
print(f"Device: {DEVICE}")
# =========================
# LOAD CHECKPOINT
# =========================
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

# =========================
# BUILD MODEL BASED ON TYPE
# =========================
if MODEL_TYPE == "2d":
    # 2D Model (hardcoded config, same as training defaults)
    enc = ResEncoder2D(
        in_ch=1,
        base_ch=32,
        num_stages=4,
        blocks_per_stage=2
    ).to(DEVICE)

    with torch.no_grad():
        dummy_input = torch.randn(1, 1, INPUT_SHAPE[2], INPUT_SHAPE[3], device=DEVICE)
        feats = enc(dummy_input)

    feat_chs = [f.shape[1] for f in feats]

    dec = ResDecoder2D(
        encoder_feature_channels=feat_chs,
        out_ch=1
    ).to(DEVICE)

else:
    # 3D Model - use config from config.py
    model_cfg = cfg["model3d_cfg"]
    stage_channels = model_cfg["stage_channels"]
    blocks_per_stage = model_cfg["blocks_per_stage"]

    enc = ResEncoder3D(
        in_ch=model_cfg["in_ch"],
        base_ch=model_cfg["base_ch"],
        num_stages=len(stage_channels),
        blocks_per_stage=blocks_per_stage,
        stage_channels=stage_channels,
        convs_per_block=model_cfg["convs_per_block"],
        norm_layer=nn.InstanceNorm3d,
        negative_slope=model_cfg["negative_slope"],
    ).to(DEVICE)

    dec = ResDecoder3D(
        encoder_feature_channels=stage_channels,
        out_ch=model_cfg["out_ch"],
        blocks_per_up=1,
        convs_per_block=model_cfg["convs_per_block"],
        norm_layer=nn.InstanceNorm3d,
        negative_slope=model_cfg["negative_slope"],
    ).to(DEVICE)

# =========================
# LOAD WEIGHTS
# =========================
enc.load_state_dict(ckpt["encoder"])
dec.load_state_dict(ckpt["decoder"])

enc.eval()
dec.eval()


# =========================
# WRAP INTO SINGLE MODEL
# =========================
class FullModel(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        feats = self.encoder(x)
        out = self.decoder(feats)
        return out


model = FullModel(enc, dec).to(DEVICE)
model.eval()

# =========================
# TRACE TO TORCHSCRIPT
# =========================
example_input = torch.randn(*INPUT_SHAPE, device=DEVICE)

with torch.no_grad():
    scripted_model = torch.jit.trace(model, example_input)


# =========================
# SAVE
# =========================
scripted_model.save(EXPORT_PATH)
print(f"Saved TorchScript model to: {EXPORT_PATH}")


# =========================
# VERIFY NUMERICAL MATCH
# =========================
with torch.no_grad():
    x_test = torch.randn(*INPUT_SHAPE, device=DEVICE)

    y_original = model(x_test)
    y_loaded = torch.jit.load(EXPORT_PATH, map_location=DEVICE)(x_test)

    max_diff = torch.max(torch.abs(y_original - y_loaded)).item()

print("Max absolute difference:", max_diff)

if max_diff < 1e-5:
    print("✔ Export verified successfully.")
else:
    print("⚠ WARNING: Output mismatch. Do NOT trust this export.")
