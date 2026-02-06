import torch

from models.unet_2d import ResEncoder2D, ResDecoder2D


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


if __name__ == "__main__":
    print("Testing ResEncoder2D...")
    test_encoder(input_size=(2, 1, 64, 64))
    print("\nTesting ResEncoder2D + ResDecoder2D...")
    test_encoder_decoder(input_size=(2, 1, 64, 64))
