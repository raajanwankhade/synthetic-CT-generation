import os

import wandb


def init_wandb(cfg, seed: int, use_amp: bool, is_3d: bool):
    os.environ["WANDB_MODE"] = cfg.logging.wandb_mode
    wandb.login()
    return wandb.init(
        project=cfg.logging.wandb_project,
        name=cfg.logging.wandb_run_name,
        config={
            "seed": seed,
            "epochs": cfg.training.epochs,
            "batch_size": cfg.dataloader.batch_size,
            "lr": cfg.training.lr,
            "patch_size": cfg.data.patch_size,
            "anatomy": cfg.data.anatomy,
            "num_adjacent_slices": cfg.data.num_adjacent_slices,
            "model": "ResNet3D_EncoderDecoder" if is_3d else "ResNet2D_EncoderDecoder",
            "scheduler": "ReduceLROnPlateau",
            "loss": "L1 + SSIM + Seg Loss",
            "amp": use_amp,
        },
        mode=cfg.logging.wandb_mode,
        resume=cfg.logging.resume_wandb,
        id=cfg.logging.id_wandb,
    )


def log_metrics(metrics: dict):
    wandb.log(metrics)
