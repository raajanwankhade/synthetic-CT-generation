import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from monai.inferers import SlidingWindowInferer
from tqdm.auto import tqdm
import config
from test_dataset_class import (
    is_patient_dir,
    make_adjacent_stack,
)

# Detect interactive environment (Jupyter/IPython)
try:
    get_ipython()  # type: ignore
    IS_INTERACTIVE = True
except NameError:
    IS_INTERACTIVE = False

# Conditional tqdm - only show progress bar in interactive environments
if IS_INTERACTIVE:
    tqdm_iter = tqdm
else:
    tqdm_iter = lambda x, **kwargs: x  # type: ignore


def _stride_to_overlap(
    patch_size: Tuple[int, ...],
    stride: Optional[Tuple[int, ...]],
) -> Tuple[float, ...]:
    """
    Convert a stride tuple to MONAI overlap values.

    overlap = 1 - stride / patch_size for each dimension.
    """
    if stride is None:
        return tuple(0.0 for _ in patch_size)

    if len(stride) != len(patch_size):
        raise ValueError("stride must have the same number of dims as patch_size")

    overlap = []
    for dim, step in zip(patch_size, stride):
        if step <= 0:
            raise ValueError("stride values must be positive")
        if step > dim:
            raise ValueError("stride values must be <= corresponding patch_size values")
        overlap.append(1.0 - (float(step) / float(dim)))
    return tuple(overlap)


def _run_sliding_window(
    model: torch.jit.ScriptModule,
    x: torch.Tensor,
    inferer: SlidingWindowInferer,
    is_3d: bool,
    channels_last: bool,
) -> np.ndarray:
    """
    Run MONAI sliding window inference and return spatial prediction.

    Args:
        model: TorchScript model
        x: batched tensor, NCHW (2D/2.5D) or NCDHW (3D)
        inferer: MONAI SlidingWindowInferer
        is_3d: True for volumetric 3D inference
        channels_last: Use channels_last memory format (should match training)

    Returns:
        Predicted patch (H,W) for 2D/2.5D or (D,H,W) for 3D
    """
    if channels_last and x.device.type != "cpu":
        if is_3d and hasattr(torch, "channels_last_3d"):
            x = x.contiguous(memory_format=torch.channels_last_3d)
        elif not is_3d:
            x = x.contiguous(memory_format=torch.channels_last)

    with torch.no_grad():
        y = inferer(x, model)

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
    stride: Optional[Tuple[int, ...]],
    output_dir: Path,
    device: str,
    num_adjacent_slices,  # int or "3d"
    channels_last: bool = False,
    use_preproc_files: bool = False,
) -> Path:
    """
    Run inference on a single patient volume.

    Supports:
      - 2D inference (num_adjacent_slices=0)
      - 2.5D inference (num_adjacent_slices=1,3,5,...)
      - 3D volumetric inference (num_adjacent_slices="3d")
    """
    if use_preproc_files:
        mod_img = sitk.ReadImage(str(patient_dir / f"{modality}_preproc.mha"))
    else:
        mod_img = sitk.ReadImage(str(patient_dir / f"{modality}.mha"))
    mod_arr = sitk.GetArrayFromImage(mod_img)  # (D, H, W)

    if mod_arr.ndim != 3:
        raise ValueError("modality image must be 3D (D,H,W)")

    depth, h, w = mod_arr.shape
    out_vol = np.zeros((depth, h, w), dtype=np.float32)

    is_3d = str(num_adjacent_slices).lower() == "3d"
    overlap = _stride_to_overlap(patch_size, stride)
    sw_batch_size = int(getattr(config, "SW_BATCH_SIZE", 1))
    if sw_batch_size <= 0:
        raise ValueError("SW_BATCH_SIZE must be a positive integer")

    inferer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        progress=IS_INTERACTIVE,
    )

    if is_3d:
        # Volumetric 3D sliding-window inference.
        if len(patch_size) != 3:
            raise ValueError("3D inference requires patch_size=(D,H,W)")
        x = torch.from_numpy(mod_arr[None, None, ...]).float().to(device)
        out_vol = _run_sliding_window(
            model=model,
            x=x,
            inferer=inferer,
            is_3d=True,
            channels_last=channels_last,
        )
    else:
        # Slice-by-slice 2D/2.5D sliding-window inference.
        num_adj = int(num_adjacent_slices)
        for z in range(depth):
            stack = make_adjacent_stack(mod_arr, z, num_adj)
            x = torch.from_numpy(stack[None, ...]).float().to(device)
            out_vol[z] = _run_sliding_window(
                model=model,
                x=x,
                inferer=inferer,
                is_3d=False,
                channels_last=channels_last,
            )

    out_img = sitk.GetImageFromArray(out_vol)
    out_img.CopyInformation(mod_img)

    # Save in structure compatible with postproc.py: output_dir/patient_name/sct.mha
    patient_output_dir = output_dir / patient_dir.name
    patient_output_dir.mkdir(parents=True, exist_ok=True)
    out_path = patient_output_dir / "sct.mha"
    sitk.WriteImage(out_img, str(out_path))
    return out_path


def main(args: argparse.Namespace) -> None:
    print("[model_test.py] Starting inference...")
    model_path = Path(config.MODEL_PATH)
    input_path = Path(config.INPUT_PATH)
    output_dir = Path(config.OUTPUT_DIR)
    preproc = config.USE_PREPROC_FILES

    print(f"[model_test.py] Model: {model_path}")
    print(f"[model_test.py] Input: {input_path}")
    print(f"[model_test.py] Output: {output_dir}")
    print(f"[model_test.py] Preprocessing: {preproc}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    device = config.DEVICE
    print(f"[model_test.py] Loading model on device: {device}")
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()

    # Determine if 3D mode
    is_3d = str(config.NUM_ADJACENT_SLICES).lower() == "3d"
    mode = "3D" if is_3d else "2D/2.5D"
    print(f"[model_test.py] Inference mode: {mode} (adjacent_slices={config.NUM_ADJACENT_SLICES})")

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

    print(f"[model_test.py] Patch size: {patch_size}, Stride: {stride}")

    patients: List[Path] = []
    if is_patient_dir(input_path, config.MODALITY):
        patients = [input_path]
    else:
        for p in sorted(input_path.iterdir()):
            if is_patient_dir(p, config.MODALITY):
                patients.append(p)

    if not patients:
        raise ValueError("No valid patient folders found")

    print(f"[model_test.py] Found {len(patients)} patient(s) to process")
    print("[model_test.py] Starting to load and process patient data...")
    channels_last = getattr(config, "CHANNELS_LAST", False)
    for patient_dir in tqdm_iter(patients, desc="Processing patients", disable=not IS_INTERACTIVE):
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
            use_preproc_files=preproc,
        )
        if not args.quiet:
            print(f"[model_test.py] Saved: {out_path}")

    print("[model_test.py] Inference completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference on patient volumes.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-patient save messages.",
    )
    main(parser.parse_args())
