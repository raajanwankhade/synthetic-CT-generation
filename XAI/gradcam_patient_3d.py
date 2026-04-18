"""Patch-wise GradCAM analysis for a single patient using the project ResUNet3D."""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

# Make project imports work even when this script is launched from XAI/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CONFIG_PATH", str(PROJECT_ROOT / "config.py"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attribution_3d_heatmaps import GradCAM3D
from inference.test_dataset_class import extract_patches_sequentially_3d
from xai_utils import (
    build_model_and_load_weights,
    default_patch_size_str,
    default_stride_str,
    load_patient_volume,
    parse_stride,
    parse_tuple3,
    predict_sct_patch,
    resolve_patient_dirs,
    save_patient_outputs,
    stitch_patches_3d,
)
from xai_config import xai_cfg


def generate_patchwise_gradcam(
    model: nn.Module,
    volume: np.ndarray,
    patch_size: Tuple[int, int, int],
    stride: Optional[Tuple[int, int, int]],
    target_layer: str,
    target_class: int,
    device: torch.device,
    stitch_blend_mode: str,
    gaussian_sigma_scale: float,
    gaussian_min_weight: float,
    compute_sct: bool,
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
    List[np.ndarray],
    List[Tuple[int, int, int]],
    Tuple[int, int, int],
]:
    """Generate full-volume GradCAM (and optional sCT) by patch-wise stitching."""
    patches, coords, full_shape = extract_patches_sequentially_3d(
        volume,
        patch_size=patch_size,
        stride=stride,
    )

    gradcam = GradCAM3D(model, target_layer=target_layer)
    cam_patches: List[np.ndarray] = []
    sct_patches: Optional[List[np.ndarray]] = [] if compute_sct else None

    for patch in tqdm(patches, desc="GradCAM patches", leave=False):
        x = torch.from_numpy(patch[None, None, ...]).float().to(device)
        # Keep gradients on input tensor while model params stay frozen to lower memory.
        x.requires_grad_(True)
        try:
            cam_patch = gradcam.generate_cam(x, target_class=target_class)
        except torch.cuda.OutOfMemoryError as e:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM during GradCAM. Try one or more: "
                "(1) deeper target_layer (e.g., encoder.stages.6.1.net.3), "
                "(2) smaller patch_size, "
                "(3) larger stride, "
                "(4) run without --save_sct."
            ) from e
        cam_patches.append(cam_patch.astype(np.float32))
        if compute_sct and sct_patches is not None:
            sct_patches.append(predict_sct_patch(model, x))

    cam_volume = stitch_patches_3d(
        cam_patches,
        coords,
        full_shape,
        stitch_blend_mode=stitch_blend_mode,
        gaussian_sigma_scale=gaussian_sigma_scale,
        gaussian_min_weight=gaussian_min_weight,
    )

    sct_volume: Optional[np.ndarray] = None
    if compute_sct and sct_patches is not None:
        sct_volume = stitch_patches_3d(
            sct_patches,
            coords,
            full_shape,
            stitch_blend_mode=stitch_blend_mode,
            gaussian_sigma_scale=gaussian_sigma_scale,
            gaussian_min_weight=gaussian_min_weight,
        ).astype(np.float32)

    cam_volume = (cam_volume - cam_volume.min()) / (cam_volume.max() - cam_volume.min() + 1e-8)
    return cam_volume.astype(np.float32), sct_volume, cam_patches, coords, full_shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch-wise GradCAM for one patient (ResUNet3D)")
    parser.add_argument(
        "--patient_dir",
        type=Path,
        required=True,
        help="Path to one patient folder or to a dataset root containing patient subfolders",
    )
    parser.add_argument(
        "--weights_path",
        type=Path,
        default=Path(str(xai_cfg["weights_path"])),
        help="Checkpoint path",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(str(xai_cfg["output_dir"])),
        help="Output directory",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default=str(xai_cfg["modality"]),
        help="Input modality name (e.g., mr)",
    )
    parser.add_argument(
        "--use_preproc_files",
        dest="use_preproc_files",
        action="store_true",
        help="Load <modality>_preproc.mha instead of <modality>.mha",
    )
    parser.add_argument(
        "--no_preproc_files",
        dest="use_preproc_files",
        action="store_false",
        help="Load <modality>.mha instead of <modality>_preproc.mha",
    )
    parser.set_defaults(use_preproc_files=bool(xai_cfg["use_preproc_files"]))
    parser.add_argument("--device", type=str, default=str(xai_cfg["device"]), help="cpu or cuda")
    parser.add_argument(
        "--patch_size",
        type=str,
        default=default_patch_size_str(),
        help="3D tuple, e.g. '(32,192,192)'",
    )
    parser.add_argument(
        "--stride",
        type=str,
        default=default_stride_str(),
        help="3D tuple or None",
    )
    parser.add_argument(
        "--target_layer",
        type=str,
        default=str(xai_cfg["target_layer"]),
        help="Model layer for GradCAM",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=int(xai_cfg["target_class"]),
        help="Output channel index",
    )
    parser.add_argument(
        "--stitch_blend_mode",
        type=str,
        choices=["gaussian", "average"],
        default=str(xai_cfg.get("stitch_blend_mode", "gaussian")),
        help="Patch stitch blending mode",
    )
    parser.add_argument(
        "--gaussian_sigma_scale",
        type=float,
        default=float(xai_cfg.get("gaussian_sigma_scale", 0.125)),
        help="Relative Gaussian sigma for weighted blending",
    )
    parser.add_argument(
        "--gaussian_min_weight",
        type=float,
        default=float(xai_cfg.get("gaussian_min_weight", 1e-3)),
        help="Minimum Gaussian weight floor to avoid near-zero edge weights",
    )
    parser.add_argument(
        "--save_gradcam",
        action="store_true",
        default=bool(xai_cfg.get("save_gradcam", False)),
        help="Save gradcam.mha",
    )
    parser.add_argument(
        "--save_overlay",
        action="store_true",
        default=bool(xai_cfg.get("save_overlay", False)),
        help="Save gradcam_overlay_mid_slice.png",
    )
    parser.add_argument(
        "--save_sct",
        action="store_true",
        default=bool(xai_cfg.get("save_sct", False)),
        help="Save generated sCT volume as sct.mha (and sct.npy if --save_npy)",
    )
    parser.add_argument(
        "--save_npy",
        action="store_true",
        default=bool(xai_cfg.get("save_npy", False)),
        help="Save .npy arrays (gradcam.npy and, if enabled, sct.npy)",
    )
    parser.add_argument(
        "--save_patch_maps",
        action="store_true",
        default=bool(xai_cfg.get("save_patch_maps", False)),
        help="Save per-patch 3D GradCAM maps (+coords) to gradcam_patches.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {args.weights_path}")

    patch_size = parse_tuple3(args.patch_size, "patch_size")
    stride = parse_stride(args.stride)
    patient_dirs = resolve_patient_dirs(args.patient_dir, args.modality, args.use_preproc_files)

    requested_device = args.device.lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    print("=" * 72)
    print("Patch-wise GradCAM (3D ResUNet) - Single/Multi Patient")
    print("=" * 72)
    print(f"Input path       : {args.patient_dir}")
    print(f"Patients found   : {len(patient_dirs)}")
    print(f"Weights          : {args.weights_path}")
    print(f"Output dir       : {args.output_dir}")
    print(f"Device           : {device}")
    print(f"Modality         : {args.modality}")
    print(f"Use preproc      : {args.use_preproc_files}")
    print(f"Patch size       : {patch_size}")
    print(f"Stride           : {stride}")
    print(f"Target layer     : {args.target_layer}")
    print(f"Target class     : {args.target_class}")
    print(f"Stitch mode      : {args.stitch_blend_mode}")
    print(f"Gaussian sigma   : {args.gaussian_sigma_scale}")
    print(f"Gaussian min wt  : {args.gaussian_min_weight}")
    print(f"Save gradcam mha : {args.save_gradcam}")
    print(f"Save overlay png : {args.save_overlay}")
    print(f"Save sCT         : {args.save_sct}")
    print(f"Save npy         : {args.save_npy}")
    print(f"Save patch maps  : {args.save_patch_maps}")

    print("\n[1/4] Building model and loading weights...")
    model = build_model_and_load_weights(args.weights_path, device)

    print("[2/4] Resolving and loading patient volumes...")
    for i, patient_dir in enumerate(tqdm(patient_dirs, desc="Patients"), start=1):
        print(f"\nPatient {i}/{len(patient_dirs)}: {patient_dir.name}")
        volume, reference_img = load_patient_volume(
            patient_dir=patient_dir,
            modality=args.modality,
            use_preproc_files=args.use_preproc_files,
        )
        print(f"      Volume shape: {volume.shape}")

        print("[3/4] Running patch-wise GradCAM...")
        cam_volume, sct_volume, cam_patches, patch_coords, _ = generate_patchwise_gradcam(
            model=model,
            volume=volume,
            patch_size=patch_size,
            stride=stride,
            target_layer=args.target_layer,
            target_class=args.target_class,
            device=device,
            stitch_blend_mode=args.stitch_blend_mode,
            gaussian_sigma_scale=args.gaussian_sigma_scale,
            gaussian_min_weight=args.gaussian_min_weight,
            compute_sct=args.save_sct,
        )

        print("[4/4] Saving outputs...")
        save_patient_outputs(
            patient_dir=patient_dir,
            output_dir=args.output_dir,
            input_volume=volume,
            cam_volume=cam_volume,
            reference_image=reference_img,
            sct_volume=sct_volume,
            save_gradcam=args.save_gradcam,
            save_overlay=args.save_overlay,
            save_sct=args.save_sct,
            save_npy=args.save_npy,
            patch_maps=cam_patches,
            patch_coords=patch_coords,
            save_patch_maps=args.save_patch_maps,
        )

        patient_output_dir = args.output_dir / patient_dir.name
        print("Saved outputs:")
        saved_any = False
        if args.save_npy:
            print(f"  {patient_output_dir / 'gradcam.npy'}")
            saved_any = True
        if args.save_gradcam:
            print(f"  {patient_output_dir / 'gradcam.mha'}")
            saved_any = True
        if args.save_sct:
            if args.save_npy:
                print(f"  {patient_output_dir / 'sct.npy'}")
            print(f"  {patient_output_dir / 'sct.mha'}")
            saved_any = True
        if args.save_overlay:
            print(f"  {patient_output_dir / 'gradcam_overlay_mid_slice.png'}")
            saved_any = True
        if args.save_patch_maps:
            print(f"  {patient_output_dir / 'gradcam_patches.npz'}")
            saved_any = True
        if not saved_any:
            print("  (none; no --save_* flags were provided)")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nDone. Processed all requested patients.")


if __name__ == "__main__":
    main()
