import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from tqdm.auto import tqdm
import config
from test_dataset_class import (
    extract_patches_sequentially,
    extract_patches_sequentially_3d,
    is_patient_dir,
    make_adjacent_stack,
    stitch_patches_array,
    stitch_patches_array_3d,
)


def _predict_patch(
    model: torch.jit.ScriptModule,
    patch: np.ndarray,
    device: str,
    is_3d: bool = False,
    channels_last: bool = False,
) -> np.ndarray:
    """
    Run inference on a single patch.

    Args:
        model: TorchScript model
        patch: 2D (H,W), 2.5D (C,H,W), or 3D (D,H,W) / (C,D,H,W) patch
        device: 'cpu' or 'cuda'
        is_3d: True for volumetric 3D inference
        channels_last: Use channels_last memory format (should match training)

    Returns:
        Predicted patch (H,W) for 2D/2.5D or (D,H,W) for 3D
    """
    if is_3d:
        # 3D: expect (D,H,W) or (C,D,H,W)
        if patch.ndim == 3:
            x = torch.from_numpy(patch[None, None, ...]).float()  # -> (1,1,D,H,W)
        elif patch.ndim == 4:
            x = torch.from_numpy(patch[None, ...]).float()  # -> (1,C,D,H,W)
        else:
            raise ValueError("3D patch must be (D,H,W) or (C,D,H,W)")
    else:
        # 2D/2.5D: expect (H,W) or (C,H,W)
        if patch.ndim == 2:
            x = torch.from_numpy(patch[None, None, ...]).float()  # -> (1,1,H,W)
        elif patch.ndim == 3:
            x = torch.from_numpy(patch[None, ...]).float()  # -> (1,C,H,W)
        else:
            raise ValueError("2D/2.5D patch must be (H,W) or (C,H,W)")

    x = x.to(device)

    # Apply channels_last memory format for GPU inference (matches training)
    if channels_last and device != "cpu":
        if is_3d and hasattr(torch, "channels_last_3d"):
            x = x.contiguous(memory_format=torch.channels_last_3d)
        elif not is_3d:
            x = x.contiguous(memory_format=torch.channels_last)

    with torch.no_grad():
        y = model(x)

    y_np = y.detach().cpu().numpy()

    # Extract spatial output
    if y_np.ndim == 5:  # (N,C,D,H,W) -> (D,H,W)
        return y_np[0, 0]
    if y_np.ndim == 4:  # (N,C,H,W) -> (H,W)
        return y_np[0, 0]
    if y_np.ndim == 3:  # (N,H,W) -> (H,W)
        return y_np[0]

    raise ValueError("model output must be NCDHW, NCHW, or NHW")


def run_patient(
    model: torch.jit.ScriptModule,
    patient_dir: Path,
    modality: str,
    patch_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    output_dir: Path,
    device: str,
    num_adjacent_slices,  # int or "3d"
    channels_last: bool = False,
) -> Path:
    """
    Run inference on a single patient volume.

    Supports:
      - 2D inference (num_adjacent_slices=0)
      - 2.5D inference (num_adjacent_slices=1,3,5,...)
      - 3D volumetric inference (num_adjacent_slices="3d")
    """
    mod_img = sitk.ReadImage(str(patient_dir / f"{modality}.mha"))
    mod_arr = sitk.GetArrayFromImage(mod_img)  # (D, H, W)

    if mod_arr.ndim != 3:
        raise ValueError("modality image must be 3D (D,H,W)")

    depth, h, w = mod_arr.shape
    out_vol = np.zeros((depth, h, w), dtype=np.float32)

    is_3d = str(num_adjacent_slices).lower() == "3d"

    if is_3d:
        # Volumetric 3D patch-based inference
        if len(patch_size) != 3:
            raise ValueError("3D inference requires patch_size=(D,H,W)")

        patches, coords, _ = extract_patches_sequentially_3d(
            mod_arr, patch_size=patch_size, stride=stride
        )
        pred_patches: List[np.ndarray] = []
        for p in patches:
            pred = _predict_patch(model, p, device, is_3d=True, channels_last=channels_last)
            pred_patches.append(pred)
        out_vol = stitch_patches_array_3d(pred_patches, coords, (depth, h, w))
    else:
        # Slice-by-slice 2D or 2.5D inference
        num_adj = int(num_adjacent_slices)
        for z in range(depth):
            stack = make_adjacent_stack(mod_arr, z, num_adj)

            patches, coords, _ = extract_patches_sequentially(
                stack, patch_size=patch_size, stride=stride
            )
            pred_patches: List[np.ndarray] = []
            for p in patches:
                pred = _predict_patch(model, p, device, is_3d=False, channels_last=channels_last)
                pred_patches.append(pred)
            out_slice = stitch_patches_array(pred_patches, coords, (h, w))
            out_vol[z] = out_slice

    out_img = sitk.GetImageFromArray(out_vol)
    out_img.CopyInformation(mod_img)

    # Save in structure compatible with postproc.py: output_dir/patient_name/sct.mha
    patient_output_dir = output_dir / patient_dir.name
    patient_output_dir.mkdir(parents=True, exist_ok=True)
    out_path = patient_output_dir / "sct.mha"
    sitk.WriteImage(out_img, str(out_path))
    return out_path


def main(args: argparse.Namespace) -> None:
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

    # Determine if 3D mode
    is_3d = str(config.NUM_ADJACENT_SLICES).lower() == "3d"

    if is_3d:
        # 3D: patch_size and stride are 3-tuples (D, H, W)
        if len(config.PATCH_SIZE) != 3:
            raise ValueError("3D inference requires PATCH_SIZE=(D,H,W)")
        patch_size = tuple(int(x) for x in config.PATCH_SIZE)
        stride = None
        if config.STRIDE is not None:
            if len(config.STRIDE) != 3:
                raise ValueError("3D inference requires STRIDE=(sd,sh,sw)")
            stride = tuple(int(x) for x in config.STRIDE)
    else:
        # 2D/2.5D: patch_size and stride are 2-tuples (H, W)
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

    channels_last = getattr(config, "CHANNELS_LAST", False)
    for patient_dir in tqdm(patients):
        out_path = run_patient(
            model=model,
            patient_dir=patient_dir,
            modality=config.MODALITY,
            patch_size=patch_size,
            stride=stride,
            output_dir=output_dir,
            device=device,
            num_adjacent_slices=config.NUM_ADJACENT_SLICES,
            channels_last=channels_last,
        )
        if not args.quiet:
            print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference on patient volumes.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-patient save messages.",
    )
    main(parser.parse_args())
