import os

cfg = {
    "run_dir": "path",
    # Data
    "root": '<Set Path>',
    "modality": "mr",
    "patch_size": (256, 256),
    "num_adjacent_slices": 0,  # pure 2D
    "cache_size": 1,
    "foreground_threshold": 0.5,
    "val_split": 0.2,

    # Dataloaders
    "batch_size": 4,
    "num_workers_train": 2,
    "num_workers_val": 2,
    "prefetch_factor": 4,

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
