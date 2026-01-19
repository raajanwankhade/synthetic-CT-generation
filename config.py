import os

cfg = {
    "run_dir": "/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/Unet2dAB/testrun",
    # Data
    "root": '/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/datasets/bhuvaneshsingla/synthrad2025-ab/versions/1/Task1',
    "modality": "mr",
    "patch_size": (256, 256),
    "num_adjacent_slices": 0,  # pure 2D
    "cache_size": 1,
    "foreground_threshold": 0.5,
    "val_split": 0.2,

    # Dataloaders
    "batch_size": 4,
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
    "torch_compile": True,
    "torch_compile_mode": "max-autotune",  # or "reduce-overhead"

    # SSIM cost control
    "ssim_downsample": 2,  # 1 = full-res, 2 = avg_pool2d by 2

    # Logging/I/O control
    "log_images_every": 20,  # epochs; set 0 to disable W&B image logging
    "save_vis_every": 20,    # epochs; set 0 to disable PNG writes

    # Training
    "epochs": 120,
    "lr": 1e-4,
    "lambda_ssim": 1000.0,
    "scheduler_patience": 5,
    "grad_clip": None,
    "sanity_steps": 1,

    # Checkpoint
    "ckpt_dir": "checkpoints",
    "ckpt_filename": "best.pth",

    # W&B
    "wandb_project": "MR_CT_Translation_ResNet",
    "wandb_run_name": "full_AB_UNET2d_run_L1_SSIM",
    "wandb_mode": os.environ.get("WANDB_MODE", "online"),

    #Figure save locations
    "Loss_Plot_name": "Loss Plot.png",
    "Visualisation_Fig": "Batch_Output.png"
}
