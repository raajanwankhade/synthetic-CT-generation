from pathlib import Path
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk
import torch

from inference import config
from inference.test_dataset_class import (
    extract_patches_sequentially,
    is_patient_dir,
    make_adjacent_stack,
    read_image_to_array,
    stitch_patches_array,
)
from image_metrics import ImageMetrics
img_metrics = ImageMetrics()


def _predict_patch(model: torch.jit.ScriptModule, patch: np.ndarray, device: str) -> np.ndarray:
    if patch.ndim == 2:
        x = torch.from_numpy(patch[None, None, ...]).float()
    elif patch.ndim == 3:
        x = torch.from_numpy(patch[None, ...]).float()
    else:
        raise ValueError("patch must be 2D or 3D")

    x = x.to(device)
    with torch.no_grad():
        y = model(x)

    y_np = y.detach().cpu().numpy()
    if y_np.ndim == 4:
        return y_np[0, 0]
    if y_np.ndim == 3:
        return y_np[0]

    raise ValueError("model output must be NCHW or NHW")


def run_patient(
    model: torch.jit.ScriptModule,
    patient_dir: Path,
    modality: str,
    patch_size: Tuple[int, int],
    stride: Tuple[int, int],
    output_dir: Path,
    device: str,
    num_adjacent_slices: int,
) -> Path:
    mod_img = sitk.ReadImage(str(patient_dir / f"{modality}.mha"))
    mod_arr = sitk.GetArrayFromImage(mod_img)  # (D, H, W)

    if mod_arr.ndim != 3:
        raise ValueError("modality image must be 3D (D,H,W)")

    depth, h, w = mod_arr.shape
    out_vol = np.zeros((depth, h, w), dtype=np.float32)

    for z in range(depth):
        stack = make_adjacent_stack(mod_arr, z, num_adjacent_slices)

        patches, coords, _ = extract_patches_sequentially(
            stack, patch_size=patch_size, stride=stride
        )
        pred_patches: List[np.ndarray] = []
        for p in patches:
            pred = _predict_patch(model, p, device)
            pred_patches.append(pred)
        out_slice = stitch_patches_array(pred_patches, coords, (h, w))
        out_vol[z] = out_slice

    out_img = sitk.GetImageFromArray(out_vol)
    out_img.CopyInformation(mod_img)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"sCT_{patient_dir.name}.mha"
    sitk.WriteImage(out_img, str(out_path))
    return out_path


def main() -> None:
    model_path = Path(config.MODEL_PATH)
    input_path = Path(config.INPUT_PATH)
    output_dir = Path(config.OUTPUT_DIR)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    device = config.DEVICE
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()

    patch_size = (int(config.PATCH_SIZE[0]), int(config.PATCH_SIZE[1]))
    stride = None
    if config.STRIDE is not None:
        stride = (int(config.STRIDE[0]), int(config.STRIDE[1]))

    patients: List[Path] = []
    if is_patient_dir(input_path, config.MODALITY):
        patients = [input_path]
    else:
        for p in sorted(input_path.iterdir()):
            if is_patient_dir(p, config.MODALITY):
                patients.append(p)

    if not patients:
        raise ValueError("No valid patient folders found")

    per_patient = []
    for patient_dir in patients:
        out_path = run_patient(
            model=model,
            patient_dir=patient_dir,
            modality=config.MODALITY,
            patch_size=patch_size,
            stride=stride,
            output_dir=output_dir,
            device=device,
            num_adjacent_slices=int(config.NUM_ADJACENT_SLICES),
        )
        print(f"Saved: {out_path}")

        gt_ct_path = patient_dir / "ct.mha"
        pred_ct_path = out_path
        mask_path = patient_dir / "mask.mha"

        gt_ct = read_image_to_array(gt_ct_path)
        pred_ct = read_image_to_array(pred_ct_path)
        mask = read_image_to_array(mask_path)

        scores = img_metrics.score_patient(gt_ct, pred_ct, mask)
        scores["patient"] = patient_dir.name
        per_patient.append(scores)

    if per_patient:
        metric_names = [k for k in per_patient[0] if k != "patient"]
        averages = {
            m: float(np.mean([p[m] for p in per_patient]))
            for m in metric_names
        }

        print("\n=== Average metrics over all patients ===")
        for k, v in averages.items():
            print(f"{k:10s}: {v:.4f}")

        # print("\n=== Per-patient metrics ===")
        # for p in per_patient:
        #     print(p)


if __name__ == "__main__":
    main()