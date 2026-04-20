import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import torch
import wandb

from config import cfg as default_cfg
from dataset_class import MRCTDataset, build_dataloaders
from discriminator import build_discriminator3d
import train_utils as tu
from UNET_Loss import LossModule
from UNET_model import build_resunet3d


SHUTDOWN_FLAG_PATH = None
from wgan_gp_trainer import train_wgan_gp
from wgan_gp_utils import _load_secrets_from_file, _select_device, seed_everything


_load_secrets_from_file()


def _plot_history(history: dict[str, list[float]], run_dir: Path, cfg: dict) -> None:
    epochs = range(1, len(history["gen_loss"]) + 1)

    plt.figure(figsize=(14, 18))

    plt.subplot(4, 1, 1)
    plt.plot(epochs, history["gen_loss"], label="Generator Loss", linewidth=2)
    plt.plot(epochs, history["val_gen_loss"], label="Val Generator Loss", linewidth=2)
    plt.plot(epochs, history["critic_loss"], label="Critic Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("WGAN-GP Training Losses")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 2)
    plt.plot(epochs, history["gen_recon_loss"], label="Generator Recon", linewidth=2)
    plt.plot(epochs, history["gen_adv_loss"], label="Generator Adversarial", linewidth=2)
    plt.plot(epochs, history["critic_gp"], label="Critic GP", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Component Loss")
    plt.title("Loss Components")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.plot(epochs, history["val_ssim"], label="SSIM", linewidth=2)
    plt.plot(epochs, history["val_psnr"], label="PSNR", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Image Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 4)
    plt.plot(epochs, history["val_dice"], label="Dice", linewidth=2)
    plt.plot(epochs, history["critic_wasserstein"], label="Wasserstein", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Validation Dice and Critic Wasserstein")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(run_dir / cfg.get("Loss_Plot_name", "Loss Plot.png")))


def main(cfg: dict) -> None:
    global SHUTDOWN_FLAG_PATH
    SHUTDOWN_FLAG_PATH = os.path.join(cfg["run_dir"], "stop.flag")

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Starting UNET WGAN-GP training")
    logger.info("=" * 60)

    seed = 42
    seed_everything(seed, cfg)

    device = _select_device()
    use_amp = device.type == "cuda"
    logger.info(f"Device: {device} | AMP: {use_amp} | CUDA devices: {torch.cuda.device_count()}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        total_vram_gb = props.total_memory / (1024 ** 3)
        logger.info(
            f"GPU: {props.name} | VRAM: {total_vram_gb:.1f} GB | CPU cores: {os.cpu_count()}"
        )

    if device.type == "cuda" and cfg.get("allow_tf32", True):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        logger.warning("torch.set_float32_matmul_precision not available")

    is_3d = str(cfg.get("num_adjacent_slices", 0)).lower() == "3d"
    if not is_3d:
        raise ValueError("UNET_WGAN_GP_Train.py requires 3D training: set num_adjacent_slices='3d'")

    run_dir = Path(cfg["run_dir"])
    tu.init_wandb(cfg, seed, use_amp, is_3d)

    dataset = MRCTDataset(
        root=cfg["root"],
        modality=cfg["modality"],
        use_preproc_files=cfg.get("use_preproc_files", False),
        use_pruned_data=cfg.get("use_pruned_data", False),
        patch_size=cfg["patch_size"],
        anatomy=cfg.get("anatomy", None),
        num_adjacent_slices=cfg["num_adjacent_slices"],
        max_retries=int(cfg.get("max_retries", 8)),
        foreground_threshold=cfg["foreground_threshold"],
        cache_size=cfg["cache_size"],
    )
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(
        f"Patch size: {dataset.patch_size} | "
        f"max_retries: {int(cfg.get('max_retries', 8))}"
    )

    sample = dataset[0]
    logger.info(
        f"Sample shapes - input: {sample['input'].shape}, "
        f"label: {sample['label'].shape}, mask: {sample['mask'].shape}"
    )
    assert sample["input"].dim() == 4, "Expected 3D sample input with shape (C, D, H, W)"

    train_loader, val_loader, train_ds, val_ds, train_patients, val_patients = build_dataloaders(
        dataset, cfg, seed, device
    )
    logger.info(f"Patients: {len(train_patients)} train | {len(val_patients)} val")
    logger.info(f"Samples: {len(train_ds)} train | {len(val_ds)} val")

    gen_cfg = cfg["model3d_cfg"]
    disc_cfg = cfg["discriminator3d_cfg"]

    encoder, decoder = build_resunet3d(gen_cfg)
    critic = build_discriminator3d(disc_cfg)
    logger.info(f"Generator config: {gen_cfg}")
    logger.info(f"Discriminator config: {disc_cfg}")

    encoder.to(device)
    decoder.to(device)
    critic.to(device)

    if cfg.get("channels_last", False) and hasattr(torch, "channels_last_3d"):
        encoder = cast(Any, encoder).to(memory_format=torch.channels_last_3d)
        decoder = cast(Any, decoder).to(memory_format=torch.channels_last_3d)
        critic = cast(Any, critic).to(memory_format=torch.channels_last_3d)

    compiled_gen = False
    compiled_critic = False
    if cfg.get("torch_compile", False):
        compile_generator = bool(cfg.get("compile_generator", True))
        compile_critic = bool(cfg.get("compile_critic", False))

        if compile_generator:
            encoder, decoder, compiled_gen = tu.maybe_compile_models(encoder, decoder, cfg, device)
        if compile_critic:
            critic, compiled_critic = tu.maybe_compile_module(critic, "critic", cfg)
        else:
            logger.info("Skipping critic torch.compile for WGAN-GP stability")

        logger.info(
            f"torch.compile status | generator: {compiled_gen} | critic: {compiled_critic}"
        )

    lambda_ssim = cfg["lambda_ssim"]
    lambda_seg = cfg.get("lambda_seg", 0.1)
    ssim_down = int(cfg.get("ssim_downsample", 1))
    loss_module = LossModule(
        device=device,
        ssim_down=ssim_down,
        ssim_weight=lambda_ssim,
        seg_weight=lambda_seg,
        return_items=False,
    )

    trained_encoder, trained_decoder, trained_critic, history = train_wgan_gp(
        encoder=encoder,
        decoder=decoder,
        critic=critic,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_module=loss_module,
        cfg=cfg,
        run_dir=run_dir,
        device=device,
        num_epochs=cfg["epochs"],
        start_epoch=cfg["start"],
        resume_path=cfg.get("resume_path"),
        grad_clip=cfg.get("grad_clip"),
        shutdown_flag_path=SHUTDOWN_FLAG_PATH,
    )

    _plot_history(history, run_dir, cfg)

    if cfg.get("wandb_mode", "online") != "disabled":
        fig = tu.visualize_batch(trained_encoder, trained_decoder, val_loader, device=device, num_samples=4)
        wandb.log({"final_batch_preview": wandb.Image(fig)})
        plt.close(fig)
        wandb.finish()

    _ = trained_critic  # keep explicit for readability; returned for downstream reuse if needed

    if tu.SHUTDOWN_REQUESTED and SHUTDOWN_FLAG_PATH and os.path.exists(SHUTDOWN_FLAG_PATH):
        logger.info("Shutdown complete. Removing stop flag.")
        os.remove(SHUTDOWN_FLAG_PATH)

    logger.info("=" * 60)
    logger.info("WGAN-GP training complete")
    logger.info("=" * 60)


def _infer_type(value):
    if value is None:
        return str
    if isinstance(value, bool):
        return str
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    return str


def _parse_value(value_str: str, default_value):
    if default_value is None:
        if value_str.lower() == "none":
            return None
        if value_str.lower() in ("true", "false"):
            return value_str.lower() == "true"
        try:
            return int(value_str)
        except ValueError:
            pass
        try:
            return float(value_str)
        except ValueError:
            pass
        return value_str

    if isinstance(default_value, bool):
        return value_str.lower() in ("true", "1", "yes")
    if isinstance(default_value, int):
        return int(value_str)
    if isinstance(default_value, float):
        return float(value_str)
    if isinstance(default_value, (list, tuple)):
        try:
            result = json.loads(value_str)
        except json.JSONDecodeError:
            result = []
            for item in value_str.split(","):
                item = item.strip()
                try:
                    result.append(int(item))
                except ValueError:
                    try:
                        result.append(float(item))
                    except ValueError:
                        result.append(item)
        return tuple(result) if isinstance(default_value, tuple) else result
    if isinstance(default_value, dict):
        return json.loads(value_str)

    if value_str.lower() in ("true", "false"):
        return value_str.lower() == "true"
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def build_argparser(cfg: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UNET WGAN-GP Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    for key, value in cfg.items():
        if key == "run_dir":
            parser.add_argument(
                f"--{key}",
                type=str,
                required=True,
                help=f"(REQUIRED) Run directory. Default from config: {value}",
            )
        elif isinstance(value, bool):
            parser.add_argument(
                f"--{key}",
                type=str,
                default=str(value),
                choices=["true", "false", "True", "False", "1", "0"],
                help=f"Default: {value}",
            )
        elif isinstance(value, (list, tuple, dict)):
            parser.add_argument(
                f"--{key}",
                type=str,
                default=json.dumps(value),
                help=f"JSON string. Default: {json.dumps(value)}",
            )
        else:
            arg_type = _infer_type(value)
            parser.add_argument(
                f"--{key}",
                type=arg_type if value is not None else str,
                default=value if value is not None else "None",
                help=f"Default: {value}",
            )

    parser.add_argument(
        "--log_file",
        type=str,
        default="train_wgan_gp.log",
        help="Log file name (default: train_wgan_gp.log)",
    )
    return parser


def update_cfg_from_args(cfg: dict, args: argparse.Namespace) -> dict:
    updated_cfg = cfg.copy()
    for key, default_value in cfg.items():
        if hasattr(args, key):
            arg_value = getattr(args, key)
            if isinstance(arg_value, str):
                updated_cfg[key] = _parse_value(arg_value, default_value)
            else:
                updated_cfg[key] = arg_value
    return updated_cfg


def save_config_to_run_dir(cfg: dict, run_dir: Path, logger: logging.Logger) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.py"

    lines = ["cfg = {\n"]
    for key, value in cfg.items():
        lines.append(f'    "{key}": {repr(value)},\n')
    lines.append("}\n")

    with open(config_path, "w") as f:
        f.writelines(lines)

    logger.info(f"Config saved to: {config_path}")
    return config_path


if __name__ == "__main__":
    parser = build_argparser(default_cfg)
    args = parser.parse_args()

    cfg = update_cfg_from_args(default_cfg, args)
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / args.log_file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w")],
    )
    logger = logging.getLogger(__name__)

    config_path = save_config_to_run_dir(cfg, run_dir, logger)
    os.environ["CONFIG_PATH"] = str(config_path)
    logger.info(f"CONFIG_PATH set to: {config_path}")

    SHUTDOWN_FLAG_PATH = os.path.join(cfg["run_dir"], "stop.flag")
    main(cfg)
