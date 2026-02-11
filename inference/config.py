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

if __name__ == "__main__":
    import argparse
    import ast
    import sys

    def tuple_or_none(value):
        if value.lower() == "none":
            return None
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, tuple):
                return parsed
        except Exception:
            pass
        raise argparse.ArgumentTypeError(
            "Must be a tuple like '(32,192,160)' or 'None'"
        )

    def slices_type(value):
        if value.lower() == "3d":
            return "3d"
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Must be integer or '3d'"
            )

    def bool_type(value):
        if value.lower() in {"true", "1", "yes"}:
            return True
        if value.lower() in {"false", "0", "no"}:
            return False
        raise argparse.ArgumentTypeError("Boolean expected")

    parser = argparse.ArgumentParser(description="Override config values")

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--modality", type=str)
    parser.add_argument("--patch_size", type=tuple_or_none)
    parser.add_argument("--stride", type=tuple_or_none)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--num_adjacent_slices", type=slices_type)
    parser.add_argument("--channels_last", type=bool_type)

    args = parser.parse_args()

    # Update globals only if provided
    g = globals()
    for key, value in vars(args).items():
        if value is not None:
            g[key.upper()] = value

    print("\nFinal configuration:\n")
    for name in [
        "MODEL_PATH",
        "INPUT_PATH",
        "MODALITY",
        "PATCH_SIZE",
        "STRIDE",
        "OUTPUT_DIR",
        "DEVICE",
        "NUM_ADJACENT_SLICES",
        "CHANNELS_LAST",
    ]:
        print(f"{name} = {g[name]}")
