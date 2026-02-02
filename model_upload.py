#Set env variables for wandb and kaggle
#download dataset
import os
import json
secrets_path = 'secrets.json'
with open(secrets_path, 'r') as f:
    secrets = json.load(f)
os.environ["KAGGLE_USERNAME"] = secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = secrets["KAGGLE_KEY"]

import kagglehub
import torch

# =========================
# USER SETTINGS
# =========================
CHECKPOINT_PATH = "/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/Unet2dTH/special_run/checkpoints/best.pth"          # <-- set this
EXPORT_PATH = "/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/Unet2dTH/special_run/checkpoints/scripted.pth"         # <-- set this
INPUT_SHAPE = (1, 1, 128, 128)              # (B, C, H, W)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# IMPORT ARCHITECTURES
# =========================
from UNET_model import ResEncoder2D, ResDecoder2D


# =========================
# LOAD CHECKPOINT
# =========================
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

# =========================
# REBUILD ENCODER
# =========================
enc = ResEncoder2D(
    in_ch=1,
    base_ch=32,
    num_stages=4,
    blocks_per_stage=2
).to(DEVICE)

with torch.no_grad():
    feats = enc(torch.randn(1, 1, 256, 256, device=DEVICE))

feat_chs = [f.shape[1] for f in feats]

# =========================
# REBUILD DECODER
# =========================
dec = ResDecoder2D(
    encoder_feature_channels=feat_chs,
    out_ch=1
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


handle = f'{secrets["KAGGLE_USERNAME"]}/UNET_TH/pytorch/special1'
local_model_dir = "/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/Unet2dTH/special_run/checkpoints/scripted.pth"

kagglehub.model_upload(handle, local_model_dir)

