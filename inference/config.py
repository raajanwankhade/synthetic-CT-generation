# Configuration for TorchScript inference

MODEL_PATH = "/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/runs/Unet3dHN/full_run2/checkpoints/scripted.pth"
INPUT_PATH = "/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/datasets/bhuvaneshsingla/synthrad2025-hn/versions/3/Task1"
MODALITY = "mr"  # "mr" or "cbct"

# Patch size:
#   2D/2.5D: (H, W) e.g., (256, 256)
#   3D:      (D, H, W) e.g., (32, 192, 160)
    # "AB": (32, 192, 224),
    # "HN": (32, 192, 160),
    # "TH": (32, 192, 192),
PATCH_SIZE = (32, 192, 160)

# Stride for overlapping patches:
#   2D/2.5D: (sh, sw) e.g., (128, 128)
#   3D:      (sd, sh, sw) e.g., (16, 96, 80)
#   None for non-overlapping
STRIDE = None

OUTPUT_DIR = "/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/inference/inference_outputs"
DEVICE = "cuda"  # "cpu" or "cuda"

# Slice mode:
#   0 for pure 2D (single slice)
#   odd values (1,3,5,...) for 2.5D (stacked adjacent slices)
#   "3d" for volumetric 3D patch-based inference
NUM_ADJACENT_SLICES = "3d"

# Use channels_last memory format for GPU inference (should match training)
CHANNELS_LAST = True
