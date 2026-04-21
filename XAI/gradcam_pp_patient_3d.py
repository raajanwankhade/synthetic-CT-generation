"""Sliding-window GradCAM++ analysis for a single patient using the project ResUNet3D."""

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

from attribution_3d_heatmaps import GradCAMPlusPlus3D
from inference.test_dataset_class import extract_patches_sequentially_3d
from xai_config import xai_cfg
from xai_utils import (
    build_sliding_window_inferer,
    build_model_and_load_weights,
    default_patch_size_str,
    default_stride_str,
    load_patient_volume,
    parse_stride,
    parse_tuple3,
    resolve_patient_dirs,
    save_patient_outputs,
)


def generate_patchwise_gradcam_pp(
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
    collect_patch_maps: bool,
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
    List[np.ndarray],
    List[Tuple[int, int, int]],
    Tuple[int, int, int],
]:
    """Generate full-volume GradCAM++ (and optional sCT) by sliding-window aggregation."""
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume (D,H,W), got shape: {volume.shape}")

    # Kept for CLI backward compatibility; MONAI inferer has no minimum-weight knob.
    _ = gaussian_min_weight

    full_shape = tuple(int(v) for v in volume.shape)
    inferer = build_sliding_window_inferer(
        patch_size=patch_size,
        stride=stride,
        stitch_blend_mode=stitch_blend_mode,
        gaussian_sigma_scale=gaussian_sigma_scale,
        sw_batch_size=1,
        progress=False,
    )

    gradcam = GradCAMPlusPlus3D(model, target_layer=target_layer)
    volume_tensor = torch.from_numpy(volume[None, None, ...]).float().to(device)

    def cam_predictor(window_batch: torch.Tensor) -> torch.Tensor:
        if window_batch.shape[0] != 1:
            raise ValueError("GradCAM++ sliding-window currently requires sw_batch_size=1")
        x = window_batch.requires_grad_(True)
        try:
            cam_patch = gradcam.generate_cam(x, target_class=target_class)
        except torch.cuda.OutOfMemoryError as e:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM during GradCAM++. Try one or more: "
                "(1) deeper target_layer (e.g., encoder.stages.6.1.net.3), "
                "(2) smaller patch_size, "
                "(3) larger stride, "
                "(4) run without --save_sct."
            ) from e
        return torch.from_numpy(cam_patch[None, None, ...]).to(
            device=x.device,
            dtype=x.dtype,
        )

    with torch.enable_grad():
        cam_pred = inferer(volume_tensor, cam_predictor)
    cam_volume = cam_pred[0, 0].detach().cpu().numpy().astype(np.float32)

    sct_volume: Optional[np.ndarray] = None
    if compute_sct:
        with torch.no_grad():
            sct_pred = inferer(volume_tensor, model)
        if sct_pred.ndim == 5:  # N,C,D,H,W
            sct_volume = sct_pred[0, 0].detach().cpu().numpy().astype(np.float32)
        elif sct_pred.ndim == 4:  # N,C,H,W
            sct_volume = sct_pred[0, 0].detach().cpu().numpy().astype(np.float32)
        elif sct_pred.ndim == 3:  # N,H,W
            sct_volume = sct_pred[0].detach().cpu().numpy().astype(np.float32)
        else:
            raise ValueError("Model output must be NCDHW, NCHW, or NHW for sCT extraction")

    cam_patches: List[np.ndarray] = []
    coords: List[Tuple[int, int, int]] = []
    if collect_patch_maps:
        patches, coords, _ = extract_patches_sequentially_3d(
            volume,
            patch_size=patch_size,
            stride=stride,
        )
        for patch in tqdm(patches, desc="GradCAM++ patch maps", leave=False):
            x = torch.from_numpy(patch[None, None, ...]).float().to(device)
            x.requires_grad_(True)
            try:
                cam_patch = gradcam.generate_cam(x, target_class=target_class)
            except torch.cuda.OutOfMemoryError as e:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA OOM during GradCAM++ patch-map export. Try one or more: "
                    "(1) deeper target_layer (e.g., encoder.stages.6.1.net.3), "
                    "(2) smaller patch_size, "
                    "(3) larger stride."
                ) from e
            cam_patches.append(cam_patch.astype(np.float32))

    cam_volume = (cam_volume - cam_volume.min()) / (cam_volume.max() - cam_volume.min() + 1e-8)
    return cam_volume.astype(np.float32), sct_volume, cam_patches, coords, full_shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sliding-window GradCAM++ for one patient (ResUNet3D)"
    )
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
        help="Model layer for GradCAM++",
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
        help="Save per-patch 3D GradCAM++ maps (+coords) to gradcam_patches.npz",
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
    print("Sliding-window GradCAM++ (3D ResUNet) - Single/Multi Patient")
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
    if args.stitch_blend_mode == "gaussian":
        print("[INFO] gaussian_min_weight is ignored in MONAI sliding-window mode.")
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

        print("[3/4] Running sliding-window GradCAM++...")
        cam_volume, sct_volume, cam_patches, patch_coords, _ = generate_patchwise_gradcam_pp(
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
            collect_patch_maps=args.save_patch_maps,
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