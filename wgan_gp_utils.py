import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import train_utils as tu


def _state_dict_for_save(module: nn.Module):
    # torch.compile wraps modules as OptimizedModule with original weights in _orig_mod.
    base_module = getattr(module, "_orig_mod", module)
    return base_module.state_dict()


def _load_state_dict_compatible(module: nn.Module, state_dict: dict, module_name: str, logger: logging.Logger) -> None:
    try:
        module.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    prefixed = all(k.startswith("_orig_mod.") for k in state_dict.keys())
    if prefixed:
        stripped_state = {k[len("_orig_mod.") :]: v for k, v in state_dict.items()}
        module.load_state_dict(stripped_state)
        logger.info(f"Loaded {module_name} weights after stripping '_orig_mod.' prefix")
        return

    compiled_state = {f"_orig_mod.{k}": v for k, v in state_dict.items()}
    module.load_state_dict(compiled_state)
    logger.info(f"Loaded {module_name} weights by applying '_orig_mod.' prefix")


def _load_secrets_from_file() -> None:
    """Populate environment variables from secrets.json when available."""
    secrets_path = Path("secrets.json")
    if not secrets_path.exists():
        return

    try:
        with open(secrets_path, "r") as f:
            secrets = json.load(f)
    except Exception:
        return

    env_map = {
        "KAGGLE_USERNAME": "KAGGLE_USERNAME",
        "KAGGLE_KEY": "KAGGLE_KEY",
        "WANDB_API_KEY": "wandb_API_KEY",
    }
    for env_key, secrets_key in env_map.items():
        value = secrets.get(secrets_key)
        if value:
            os.environ[env_key] = value


def seed_everything(seed: int = 42, cfg: dict | None = None) -> None:
    if cfg is None:
        cfg = {}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(cfg.get("deterministic", False))
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if not (0 <= local_rank < torch.cuda.device_count()):
            local_rank = 0
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _run_generator(encoder: nn.Module, decoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return decoder(encoder(x))


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(enabled)


def _build_critic_optimizer(
    critic: nn.Module,
    device: torch.device,
    lr: float,
    betas: tuple[float, float],
) -> torch.optim.Optimizer:
    try:
        return torch.optim.AdamW(
            critic.parameters(),
            lr=lr,
            betas=betas,
            fused=(device.type == "cuda"),
        )
    except TypeError:
        return torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)


def _gradient_penalty(
    critic: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
) -> torch.Tensor:
    if real_samples.shape != fake_samples.shape:
        raise ValueError(
            "real_samples and fake_samples must match in shape, got "
            f"{tuple(real_samples.shape)} vs {tuple(fake_samples.shape)}"
        )

    batch_size = real_samples.shape[0]
    alpha_shape = [batch_size] + [1] * (real_samples.ndim - 1)
    alpha = torch.rand(alpha_shape, device=real_samples.device, dtype=real_samples.dtype)

    interpolates = alpha * real_samples + (1.0 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    # WGAN-GP needs higher-order gradients via create_graph=True.
    # For compiled critics this path can hit Triton runtime issues; use eager module.
    critic_for_gp = getattr(critic, "_orig_mod", critic)
    interp_logits = critic_for_gp(interpolates)
    grad_outputs = torch.ones_like(interp_logits)

    gradients = torch.autograd.grad(
        outputs=interp_logits,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.reshape(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()


def _save_checkpoint(
    ckpt_path: Path,
    epoch: int,
    encoder: nn.Module,
    decoder: nn.Module,
    critic: nn.Module,
    gen_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    gen_scheduler,
    critic_scheduler,
    scaler,
    best_val_loss: float,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "encoder": _state_dict_for_save(encoder),
            "decoder": _state_dict_for_save(decoder),
            "critic": _state_dict_for_save(critic),
            "gen_optimizer": gen_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "gen_scheduler": gen_scheduler.state_dict(),
            "critic_scheduler": critic_scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_loss": best_val_loss,
        },
        ckpt_path,
    )


def _resume_from_checkpoint(
    resume_path: str,
    encoder: nn.Module,
    decoder: nn.Module,
    critic: nn.Module,
    gen_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    gen_scheduler,
    critic_scheduler,
    scaler,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[int, float]:
    logger.info(f"Resuming from checkpoint: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)

    _load_state_dict_compatible(encoder, ckpt["encoder"], "encoder", logger)
    _load_state_dict_compatible(decoder, ckpt["decoder"], "decoder", logger)
    _load_state_dict_compatible(critic, ckpt["critic"], "critic", logger)

    if "gen_optimizer" in ckpt:
        gen_optimizer.load_state_dict(ckpt["gen_optimizer"])
    if "critic_optimizer" in ckpt:
        critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
    if "gen_scheduler" in ckpt:
        gen_scheduler.load_state_dict(ckpt["gen_scheduler"])
    if "critic_scheduler" in ckpt:
        critic_scheduler.load_state_dict(ckpt["critic_scheduler"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    logger.info(f"Resumed at epoch={start_epoch}, best_val_loss={best_val_loss:.6f}")
    return start_epoch, best_val_loss


def _format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _check_shutdown_requested(shutdown_flag_path: str | None, logger: logging.Logger) -> bool:
    if not tu.SHUTDOWN_REQUESTED and shutdown_flag_path and os.path.exists(shutdown_flag_path):
        logger.warning("Shutdown flag detected. Training will stop after current epoch.")
        tu.SHUTDOWN_REQUESTED = True
    return tu.SHUTDOWN_REQUESTED
