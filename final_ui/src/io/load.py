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
            ct, sct, mri, gradcam (optional)
            shape, spacing
    """
    case_dir = Path(case_dir)

    paths = {
        "ct": case_dir / "ct_preproc.mha",
        "sct": case_dir / "sct.mha",
        "mri": case_dir / "mr_preproc.mha",
        # "gradcam": case_dir / "gradcam.mha",
    }

    data = {}
    spacings = {}

    # ---- Load required volumes ----
    for key in ["ct", "sct", "mri"]:
        path = paths[key]
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        arr, spacing = load_mha(path)
        data[key] = arr
        spacings[key] = spacing

    # ---- Load optional gradcam ----
    # gradcam_path = paths["gradcam"]
    # if gradcam_path.exists():
    #     arr, spacing = load_mha(gradcam_path)
    #     data["gradcam"] = arr
    #     spacings["gradcam"] = spacing
    # else:
    #     data["gradcam"] = None

    # --- Shape validation (ignore gradcam if None) ---
    base_shapes = [data[k].shape for k in ["ct", "sct", "mri"]]
    if not all(s == base_shapes[0] for s in base_shapes):
        raise ValueError(f"Shape mismatch (ct/sct/mri): {base_shapes}")

    # If gradcam exists, validate separately
    # if data["gradcam"] is not None:
    #     if data["gradcam"].shape != base_shapes[0]:
    #         print("⚠️ Warning: gradcam shape mismatch, ignoring gradcam")
    #         data["gradcam"] = None

    # --- Spacing check (log only) ---
    unique_spacings = set(tuple(spacings[k]) for k in spacings if spacings[k] is not None)

    if len(unique_spacings) > 1:
        print("⚠️ Warning: Different spacings detected:")
        for k, v in spacings.items():
            print(f"{k}: {v}")

    return {
        "ct": data["ct"],
        "sct": data["sct"],
        "mri": data["mri"],
        # "gradcam": data["gradcam"],  # can be None
        "shape": base_shapes[0],
        "spacing": spacings["ct"],  # assume consistent
    }