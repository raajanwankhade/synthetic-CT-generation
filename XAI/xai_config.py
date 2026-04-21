"""Centralized configuration for sliding-window GradCAM patient analysis."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


xai_cfg = {
    # Checkpoint and outputs
    "weights_path": str(
        PROJECT_ROOT
        / "ablations/Unet3D/HN/preproc_l1_ssim_dice/run1/checkpoints/best.pth"
    ),
    "output_dir": str(PROJECT_ROOT / "XAI/gradcam_outputs"),

    # Input/data behavior
    "modality": "mr",
    "use_preproc_files": True,

    # Inference/GradCAM behavior
    "device": "cuda",
    # Smaller patches reduce GradCAM backward memory on MIG 1g.24gb.
    "patch_size": (32, 192, 192),  # (D,H,W) for 3D; use (H,W) for 2D/2.5D
    # 50% overlap reduces visible patch seams in stitched CAM volumes.
    "stride": (16, 96, 96),
    # Deeper layer is more memory friendly; switch to encoder.stages.4.1.net.3 if GPU allows.
    "target_layer": "encoder.stages.6.1.net.3",
    "target_class": 0,
    "freeze_model_params_for_gradcam": True,
    "stitch_blend_mode": "gaussian",  # 'gaussian' or 'average'
    "gaussian_sigma_scale": 0.125,
    "gaussian_min_weight": 1e-3,
    # Save controls (all disabled by default; enable via CLI --save_* flags).
    "save_gradcam": False,
    "save_overlay": False,
    "save_sct": False,
    "save_npy": False,
    "save_patch_maps": False,

    # Which model block from root config.py to use for model construction
    "model_cfg_key": "model3d_cfg",
}
