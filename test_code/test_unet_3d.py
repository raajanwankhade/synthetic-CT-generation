import torch

from config import cfg
from models.unet_3d import ResEncoder3D, ResDecoder3D


def test_encoder_decoder_3d(input_size=(2, 1, 40, 128, 128)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg.model3d
    enc = ResEncoder3D(
        in_ch=model_cfg.in_ch,
        base_ch=model_cfg.base_ch,
        num_stages=len(model_cfg.stage_channels) or 4,
        blocks_per_stage=model_cfg.blocks_per_stage,
        stage_channels=model_cfg.stage_channels,
        convs_per_block=model_cfg.convs_per_block,
        norm_layer=torch.nn.InstanceNorm3d,
        negative_slope=model_cfg.negative_slope,
    ).to(device)
    dummy = torch.randn(*input_size, device=device)
    feats = enc(dummy)
    feat_chs = [f.shape[1] for f in feats]
    print("Encoder feature channels (3D):", feat_chs)
    for i, f in enumerate(feats):
        print(f"feat[{i}] shape:", f.shape)

    dec = ResDecoder3D(
        encoder_feature_channels=feat_chs,
        out_ch=model_cfg.out_ch,
        convs_per_block=model_cfg.convs_per_block,
        norm_layer=torch.nn.InstanceNorm3d,
        negative_slope=model_cfg.negative_slope,
    ).to(device)
    out = dec(enc(torch.randn(*input_size, device=device)))
    print("Output shape (3D):", out.shape)
    return out


if __name__ == "__main__":
    print("Testing ResEncoder3D + ResDecoder3D...")
    test_encoder_decoder_3d(input_size=(2, 1, 40, 128, 128))
