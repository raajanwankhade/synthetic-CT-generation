import argparse
import torch
import torch.nn as nn
from typing import List

from config import cfg
from UNET_model import (
    ResEncoder2D, ResDecoder2D,
    ResEncoder3D, ResDecoder3D
)

# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Export model to TorchScript")
    parser.add_argument("--model_type", required=True, choices=["2d", "3d"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--export_path", required=True)

    parser.add_argument("--input_shape_2d", type=int, nargs=4,
                        default=[1, 1, 256, 256])
    parser.add_argument("--input_shape_3d", type=int, nargs=5,
                        default=[1, 1, 32, 192, 160])

    return parser.parse_args()


# =========================
# MODEL WRAPPER
# =========================
class FullModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: List[torch.Tensor] = self.encoder(x)
        return self.decoder(feats)


# =========================
# MAIN
# =========================
def main():

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == "2d":
        input_shape = tuple(args.input_shape_2d)
    else:
        input_shape = tuple(args.input_shape_3d)

    print("Model type:", args.model_type)
    print("Input shape:", input_shape)
    print("Device:", device)

    # -------------------------
    # Load checkpoint
    # -------------------------
    ckpt = torch.load(args.checkpoint, map_location=device)

    # -------------------------
    # Build model
    # -------------------------
    if args.model_type == "2d":

        enc = ResEncoder2D(
            in_ch=1,
            base_ch=32,
            num_stages=4,
            blocks_per_stage=2
        ).to(device)

        with torch.no_grad():
            dummy = torch.randn(*input_shape, device=device)
            feats = enc(dummy)

        feat_chs = [f.shape[1] for f in feats]

        dec = ResDecoder2D(
            encoder_feature_channels=feat_chs,
            out_ch=1
        ).to(device)

    else:

        mcfg = cfg["model3d_cfg"]

        enc = ResEncoder3D(
            in_ch=mcfg["in_ch"],
            base_ch=mcfg["base_ch"],
            num_stages=len(mcfg["stage_channels"]),
            blocks_per_stage=mcfg["blocks_per_stage"],
            stage_channels=mcfg["stage_channels"],
            convs_per_block=mcfg["convs_per_block"],
            norm_layer=nn.InstanceNorm3d,
            negative_slope=mcfg["negative_slope"],
        ).to(device)

        dec = ResDecoder3D(
            encoder_feature_channels=mcfg["stage_channels"],
            out_ch=mcfg["out_ch"],
            blocks_per_up=1,
            convs_per_block=mcfg["convs_per_block"],
            norm_layer=nn.InstanceNorm3d,
            negative_slope=mcfg["negative_slope"],
        ).to(device)

    # -------------------------
    # Load weights
    # -------------------------
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])

    enc.eval()
    dec.eval()

    model = FullModel(enc, dec).to(device).eval()

    # -------------------------
    # Script model
    # -------------------------
    print("Scripting model...")
    scripted = torch.jit.script(model)

    scripted.save(args.export_path)
    print("Saved to:", args.export_path)

    # -------------------------
    # Verify export
    # -------------------------
    print("Verifying export...")

    with torch.no_grad():

        x = torch.randn(*input_shape, device=device)

        y_ref = model(x)

        loaded = torch.jit.load(args.export_path, map_location=device)
        y_ts = loaded(x)

        max_diff = torch.max(torch.abs(y_ref - y_ts)).item()

    print("Max abs difference:", max_diff)

    if max_diff < 1e-5:
        print("✔ Export verified")
    else:
        print("⚠ Output mismatch")


if __name__ == "__main__":
    main()
