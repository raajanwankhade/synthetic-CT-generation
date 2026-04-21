"""Reusable helpers for XAI scripts in this project."""

import ast
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer

from UNET_model import build_resunet3d
from config import cfg
from inference.test_dataset_class import stitch_patches_array_3d
from xai_config import xai_cfg


def default_patch_size_str() -> str:
    return str(tuple(int(x) for x in xai_cfg["patch_size"]))


def default_stride_str() -> str:
    stride = xai_cfg.get("stride")
    if stride is None:
        return "None"
    return str(tuple(int(x) for x in stride))


def parse_tuple3(value: str, name: str) -> Tuple[int, int, int]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, tuple) or len(parsed) != 3:
        raise ValueError(f"{name} must be a 3-tuple, got: {value}")
    out = tuple(int(v) for v in parsed)
    if any(v <= 0 for v in out):
        raise ValueError(f"{name} values must be positive, got: {out}")
    return out


def parse_stride(value: str) -> Optional[Tuple[int, int, int]]:
    if value.lower() == "none":
        return None
    return parse_tuple3(value, "stride")


def stride_to_overlap_3d(
    patch_size: Tuple[int, int, int],
    stride: Optional[Tuple[int, int, int]],
) -> Tuple[float, float, float]:
    """Convert explicit 3D stride to MONAI overlap values.

    overlap = 1 - stride / patch_size per dimension.
    """
    if stride is None:
        return (0.0, 0.0, 0.0)

    if len(stride) != 3:
        raise ValueError(f"stride must be a 3-tuple, got: {stride}")

    overlap: List[float] = []
    for dim, step in zip(patch_size, stride):
        if step <= 0:
            raise ValueError(f"stride values must be positive, got: {stride}")
        if step > dim:
            raise ValueError(
                f"stride must be <= patch_size per dim, got stride={stride}, patch_size={patch_size}"
            )
        overlap.append(1.0 - (float(step) / float(dim)))
    return (float(overlap[0]), float(overlap[1]), float(overlap[2]))


def build_sliding_window_inferer(
    patch_size: Tuple[int, int, int],
    stride: Optional[Tuple[int, int, int]],
    stitch_blend_mode: str,
    gaussian_sigma_scale: float,
    sw_batch_size: int = 1,
    progress: bool = False,
) -> SlidingWindowInferer:
    """Build a MONAI sliding-window inferer matching XAI config semantics."""
    if sw_batch_size <= 0:
        raise ValueError(f"sw_batch_size must be positive, got: {sw_batch_size}")

    overlap_tuple = stride_to_overlap_3d(patch_size, stride)
    overlap_arg: float | Tuple[float, float, float]
    if overlap_tuple[0] == overlap_tuple[1] == overlap_tuple[2]:
        overlap_arg = overlap_tuple[0]
    else:
        overlap_arg = overlap_tuple

    mode = "gaussian" if stitch_blend_mode == "gaussian" else "constant"

    kwargs = {
        "roi_size": patch_size,
        "sw_batch_size": sw_batch_size,
        "overlap": overlap_arg,
        "mode": mode,
        "progress": progress,
    }
    if mode == "gaussian":
        kwargs["sigma_scale"] = float(gaussian_sigma_scale)

    try:
        return SlidingWindowInferer(**kwargs)
    except TypeError:
        if isinstance(overlap_arg, tuple):
            # Older MONAI versions may only accept scalar overlap.
            kwargs["overlap"] = float(max(overlap_arg))
            return SlidingWindowInferer(**kwargs)
        raise


def get_modality_filename(modality: str, use_preproc_files: bool) -> str:
    return f"{modality}_preproc.mha" if use_preproc_files else f"{modality}.mha"


def is_patient_input_dir(path: Path, modality: str, use_preproc_files: bool) -> bool:
    if not path.is_dir():
        return False
    return (path / get_modality_filename(modality, use_preproc_files)).exists()


def resolve_patient_dirs(input_path: Path, modality: str, use_preproc_files: bool) -> List[Path]:
    """Resolve input path to a single patient folder or all patient subfolders."""
    if not input_path.exists() or not input_path.is_dir():
        raise NotADirectoryError(f"Invalid input directory: {input_path}")

    if is_patient_input_dir(input_path, modality, use_preproc_files):
        return [input_path]

    patients = [
        p
        for p in sorted(input_path.iterdir())
        if is_patient_input_dir(p, modality, use_preproc_files)
    ]
    if not patients:
        expected = get_modality_filename(modality, use_preproc_files)
        raise ValueError(
            f"No patient folders found under {input_path} (expected each folder to contain {expected})"
        )
    return patients


def build_model_and_load_weights(weights_path: Path, device: torch.device) -> nn.Module:
    """Build ResUNet3D from config and load training checkpoint weights."""
    model_cfg_key = str(xai_cfg.get("model_cfg_key", "model3d_cfg"))
    model_cfg = cfg[model_cfg_key]
    encoder, decoder = build_resunet3d(model_cfg)

    checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)
    if "encoder" not in checkpoint or "decoder" not in checkpoint:
        raise KeyError(
            "Checkpoint must contain 'encoder' and 'decoder' keys from UNET_Train.py output"
        )

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    model = FullModel(encoder, decoder).to(device)
    model.eval()
    if bool(xai_cfg.get("freeze_model_params_for_gradcam", True)):
        for p in model.parameters():
            p.requires_grad_(False)
    return model


class FullModel(nn.Module):
    """Single forward model that combines encoder and decoder."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: List[torch.Tensor] = self.encoder(x)
        return self.decoder(feats)


def load_patient_volume(patient_dir: Path, modality: str, use_preproc_files: bool) -> Tuple[np.ndarray, sitk.Image]:
    """Load one patient modality volume as a NumPy array (D, H, W)."""
    volume_path = patient_dir / get_modality_filename(modality, use_preproc_files)
    if not volume_path.exists():
        raise FileNotFoundError(f"Could not find modality file: {volume_path}")

    image = sitk.ReadImage(str(volume_path))
    array = sitk.GetArrayFromImage(image).astype(np.float32)

    if array.ndim != 3:
        raise ValueError(f"Expected 3D volume (D,H,W), got shape: {array.shape}")

    return array, image


def build_gaussian_weight_3d(
    patch_shape: Tuple[int, int, int],
    sigma_scale: float,
    min_weight: float,
) -> np.ndarray:
    """Create a separable 3D Gaussian weight map for patch blending."""
    if sigma_scale <= 0:
        raise ValueError(f"gaussian sigma_scale must be > 0, got {sigma_scale}")
    if min_weight <= 0:
        raise ValueError(f"gaussian min_weight must be > 0, got {min_weight}")

    def gauss_1d(length: int) -> np.ndarray:
        center = (length - 1) / 2.0
        sigma = max(length * sigma_scale, 1e-6)
        x = np.arange(length, dtype=np.float32)
        return np.exp(-0.5 * ((x - center) / sigma) ** 2).astype(np.float32)

    pd, ph, pw = patch_shape
    gz = gauss_1d(pd)
    gy = gauss_1d(ph)
    gx = gauss_1d(pw)

    weight = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
    weight /= max(float(weight.max()), 1e-8)
    weight = np.maximum(weight, min_weight)
    return weight.astype(np.float32)


def stitch_patches_array_3d_gaussian(
    patches: List[np.ndarray],
    coords: List[Tuple[int, int, int]],
    output_shape: Tuple[int, int, int],
    sigma_scale: float,
    min_weight: float,
) -> np.ndarray:
    """Stitch 3D patches using Gaussian weighted blending in overlap regions."""
    if len(patches) != len(coords):
        raise ValueError("patches and coords must have the same length")
    if not patches:
        raise ValueError("patches is empty")

    d, h, w = output_shape
    out = np.zeros((d, h, w), dtype=np.float32)
    weight_sum = np.zeros((d, h, w), dtype=np.float32)

    weight_cache: dict[Tuple[int, int, int], np.ndarray] = {}

    for patch, (z0, y0, x0) in zip(patches, coords):
        if patch.ndim == 4 and patch.shape[0] == 1:
            patch = patch[0]
        if patch.ndim != 3:
            raise ValueError("patch must be 3D (D,H,W) or (1,D,H,W)")

        pd, ph, pw = patch.shape
        shape_key = (pd, ph, pw)
        if shape_key not in weight_cache:
            weight_cache[shape_key] = build_gaussian_weight_3d(
                patch_shape=shape_key,
                sigma_scale=sigma_scale,
                min_weight=min_weight,
            )
        patch_weight = weight_cache[shape_key]

        out[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] += patch * patch_weight
        weight_sum[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] += patch_weight

    return out / np.maximum(weight_sum, 1e-8)


def stitch_patches_3d(
    patches: List[np.ndarray],
    coords: List[Tuple[int, int, int]],
    output_shape: Tuple[int, int, int],
    stitch_blend_mode: str,
    gaussian_sigma_scale: float,
    gaussian_min_weight: float,
) -> np.ndarray:
    """Stitch 3D patches using configured blending mode."""
    if stitch_blend_mode == "gaussian":
        return stitch_patches_array_3d_gaussian(
            patches,
            coords,
            output_shape,
            sigma_scale=gaussian_sigma_scale,
            min_weight=gaussian_min_weight,
        )
    if stitch_blend_mode == "average":
        return stitch_patches_array_3d(patches, coords, output_shape)
    raise ValueError(
        f"Unknown stitch_blend_mode '{stitch_blend_mode}'. Use 'gaussian' or 'average'."
    )


def predict_sct_patch(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Run model forward and return a single-channel sCT patch as numpy array."""
    with torch.no_grad():
        y = model(x)

    if y.ndim == 5:  # N,C,D,H,W
        return y[0, 0].detach().cpu().numpy().astype(np.float32)
    if y.ndim == 4:  # N,C,H,W
        return y[0, 0].detach().cpu().numpy().astype(np.float32)
    if y.ndim == 3:  # N,H,W
        return y[0].detach().cpu().numpy().astype(np.float32)
    raise ValueError("Model output must be NCDHW, NCHW, or NHW for sCT extraction")


def save_patient_outputs(
    patient_dir: Path,
    output_dir: Path,
    input_volume: np.ndarray,
    cam_volume: np.ndarray,
    reference_image: sitk.Image,
    sct_volume: Optional[np.ndarray] = None,
    save_gradcam: bool = False,
    save_overlay: bool = False,
    save_sct: bool = False,
    save_npy: bool = False,
    patch_maps: Optional[List[np.ndarray]] = None,
    patch_coords: Optional[List[Tuple[int, int, int]]] = None,
    save_patch_maps: bool = False,
) -> None:
    """Save XAI outputs for one patient folder."""
    if not any([save_gradcam, save_overlay, save_sct, save_npy, save_patch_maps]):
        return

    patient_output_dir = output_dir / patient_dir.name
    patient_output_dir.mkdir(parents=True, exist_ok=True)

    if save_npy:
        np.save(patient_output_dir / "gradcam.npy", cam_volume)

    if save_gradcam:
        cam_img = sitk.GetImageFromArray(cam_volume)
        cam_img.CopyInformation(reference_image)
        sitk.WriteImage(cam_img, str(patient_output_dir / "gradcam.mha"))

    if save_sct:
        if sct_volume is None:
            raise ValueError("sct_volume is required when save_sct=True")
        if save_npy:
            np.save(patient_output_dir / "sct.npy", sct_volume)
        sct_img = sitk.GetImageFromArray(sct_volume)
        sct_img.CopyInformation(reference_image)
        sitk.WriteImage(sct_img, str(patient_output_dir / "sct.mha"))

    if save_patch_maps:
        if patch_maps is None or patch_coords is None:
            raise ValueError("patch_maps and patch_coords are required when save_patch_maps=True")
        cams_array = np.stack(patch_maps, axis=0).astype(np.float32)
        coords_array = np.asarray(patch_coords, dtype=np.int32)
        np.savez_compressed(
            patient_output_dir / "gradcam_patches.npz",
            cams=cams_array,
            coords=coords_array,
            patch_shape=np.asarray(cams_array.shape[1:], dtype=np.int32),
            volume_shape=np.asarray(cam_volume.shape, dtype=np.int32),
        )

    if save_overlay:
        mid = cam_volume.shape[0] // 2
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(input_volume[mid], cmap="gray")
        axes[0].set_title(f"Input ({mid})")
        axes[0].axis("off")

        hm = axes[1].imshow(cam_volume[mid], cmap="jet")
        axes[1].set_title(f"GradCAM ({mid})")
        axes[1].axis("off")
        plt.colorbar(hm, ax=axes[1], fraction=0.046)

        axes[2].imshow(input_volume[mid], cmap="gray")
        axes[2].imshow(cam_volume[mid], cmap="jet", alpha=0.5)
        axes[2].set_title(f"Overlay ({mid})")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(patient_output_dir / "gradcam_overlay_mid_slice.png", dpi=150, bbox_inches="tight")
        plt.close(fig)