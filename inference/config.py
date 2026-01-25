# Configuration for TorchScript inference

MODEL_PATH = ""
INPUT_PATH = ""
MODALITY = "mr"  # "mr" or "cbct"
PATCH_SIZE = (256, 256)  # (H, W)
STRIDE = None  # e.g., (128, 128) or None for non-overlap
OUTPUT_DIR = "inference_outputs"
DEVICE = "cpu"  # "cpu" or "cuda"
NUM_ADJACENT_SLICES = 0  # 0 for 2D, or odd values (1,3,5,...)
