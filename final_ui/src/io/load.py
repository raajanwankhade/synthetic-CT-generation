import SimpleITK as sitk
import numpy as np
from pathlib import Path


def load_mha(path):
    """
    Load a .mha file and return numpy array + spacing
    """
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (Z, H, W)
    spacing = img.GetSpacing()         # (x, y, z)
    return arr, spacing


def load_case(case_dir):
    """
    Load all volumes for a case.

    Returns:
        dict with:
            ct, sct, mri, gradcam
            spacing
    """
    case_dir = Path(case_dir)

    paths = {
        "ct": case_dir / "ct_preproc.mha",
        "sct": case_dir / "sct.mha",
        "mri": case_dir / "mr_preproc.mha",
        "gradcam": case_dir / "gradcam.mha",
    }

    data = {}
    spacings = {}

    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        arr, spacing = load_mha(path)
        data[key] = arr
        spacings[key] = spacing

    # --- Shape validation ---
    shapes = [v.shape for v in data.values()]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError(f"Shape mismatch: {shapes}")

    # --- Spacing check (log only) ---
    # NOTE: SITK gives spacing as (x,y,z), but array is (z,y,x)
    unique_spacings = set(tuple(s) for s in spacings.values())

    if len(unique_spacings) > 1:
        print("⚠️ Warning: Different spacings detected:")
        for k, v in spacings.items():
            print(f"{k}: {v}")

    return {
        "ct": data["ct"],
        "sct": data["sct"],
        "mri": data["mri"],
        "gradcam": data["gradcam"],
        "shape": shapes[0],
        "spacing": spacings["ct"],  # assume consistent
    }