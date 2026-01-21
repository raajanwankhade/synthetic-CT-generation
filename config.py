import os

cfg = {
    "run_dir": "/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/Unet2dHN/full_run",
    # Data
    "root": '/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/datasets/bhuvaneshsingla/synthrad2025-hn/versions/1/Task1',
    "modality": "mr",
    "patch_size": (256, 256),
    "num_adjacent_slices": 0,  # pure 2D
    "cache_size": 1,
    "foreground_threshold": 0.5,
    "val_split": 0.2,

    # Dataloaders
    "batch_size": 32,
    "num_workers_train": 4,
    "num_workers_val": 4,
    "prefetch_factor": 4,

    # Perf / determinism (HPC training typically prefers speed)
    "deterministic": False,
    "cudnn_benchmark": True,
    "allow_tf32": True,

    # AMP on H100: prefer BF16
    "amp_dtype": "bf16",  # "bf16" or "fp16"
    "channels_last": True,
    "torch_compile": False,
    "torch_compile_mode": "reduce-overhead",  # or "max-autotune"

    # SSIM cost control
    "ssim_downsample": 2,  # 1 = full-res, 2 = avg_pool2d by 2

    # Logging/I/O control
    "log_images_every": 5,  # epochs; set 0 to disable W&B image logging
    "save_vis_every": 5,    # epochs; set 0 to disable PNG writes

    # Training
    "start":26,
    "epochs": 40,
    "lr": 1e-3,
    "lambda_ssim": 1000.0,
    "scheduler_patience": 10,
    "grad_clip": None,
    "sanity_steps": 1,
    "resume_path": None,

    # Checkpoint
    "ckpt_dir": "checkpoints",
    "ckpt_filename": "best.pth",

    # W&B
    "wandb_mode": "online",  # "online", "offline", or "disabled"
    "wandb_project": "MR_CT_Translation_ResNet",
    "wandb_run_name": "full_run_HN_UNET2d_run_L1_SSIM",
    "id_wandb":None,
    "resume_wandb":None,

    #Figure save locations
    "Loss_Plot_name": "Loss Plot.png",
    "Visualisation_Fig": "Batch_Output.png"
}
