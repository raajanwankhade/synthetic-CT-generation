import contextlib
from pathlib import Path
import torch
from tqdm.auto import tqdm
from monai.networks.nets import UNet
from monai.losses import DiceLoss

from config import cfg
from dataset_class import MRCTDataset, build_dataloaders

# ================= CONFIGURATION =================
MODEL_SAVE_NAME = "frozen_teacher.pth"
DEFAULT_SEED = 42
# =================================================

def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"Using device: {device}")

    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", False))
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("allow_tf32", False))

    run_dir = Path(cfg["run_dir"])
    ckpt_dir = run_dir / cfg["ckpt_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = ckpt_dir / MODEL_SAVE_NAME

    def teacher_transform(sample):
        image = sample["label"]  # CT
        label = sample["mask"]   # mask

        label = (label > 0.5).float()

        return {"image": image, "label": label}

    dataset = MRCTDataset(
        root=cfg["root"],
        modality=cfg["modality"],
        patch_size=cfg["patch_size"],
        anatomy=cfg["anatomy"],
        num_adjacent_slices=cfg["num_adjacent_slices"],
        foreground_threshold=cfg["foreground_threshold"],
        cache_size=cfg["cache_size"],
        transform=teacher_transform,
    )

    train_loader, val_loader, _, _, _, _ = build_dataloaders(
        dataset=dataset,
        cfg=cfg,
        seed=DEFAULT_SEED,
        device=device,
    )

    spatial_dims = 3 if dataset.is_3d else 2
    in_channels = 1 if dataset.is_3d else dataset.num_adjacent

    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="instance",
    ).to(device)

    if cfg.get("channels_last", False):
        if dataset.is_3d and hasattr(torch, "channels_last_3d"):
            model = model.to(memory_format=torch.channels_last_3d)
        else:
            model = model.to(memory_format=torch.channels_last)

    if cfg.get("torch_compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model, mode=cfg.get("torch_compile_mode", "reduce-overhead"))

    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    amp_dtype = torch.bfloat16 if str(cfg.get("amp_dtype", "bf16")).lower() == "bf16" else torch.float16
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    def autocast_context():
        return torch.cuda.amp.autocast(dtype=amp_dtype) if use_amp else contextlib.nullcontext()

    print("Starting Training...")
    best_val_loss = float("inf")
    patience = int(cfg.get("scheduler_patience", 10))
    no_improve = 0

    epoch_pbar = tqdm(range(cfg["epochs"]), desc="Epochs", unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0.0
        step = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{cfg['epochs']}",
            unit="batch",
            leave=False,
        )

        for batch in train_pbar:
            step += 1
            inputs, labels = batch["image"].to(device), batch["label"].to(device)

            if cfg.get("channels_last", False):
                if dataset.is_3d and hasattr(torch, "channels_last_3d"):
                    inputs = inputs.to(memory_format=torch.channels_last_3d)
                    labels = labels.to(memory_format=torch.channels_last_3d)
                else:
                    inputs = inputs.to(memory_format=torch.channels_last)
                    labels = labels.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += float(loss.item())
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = epoch_loss / max(step, 1)
        epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            val_pbar = tqdm(
                val_loader,
                desc=f"Val {epoch + 1}/{cfg['epochs']}",
                unit="batch",
                leave=False,
            )
            for batch in val_pbar:
                val_steps += 1
                inputs, labels = batch["image"].to(device), batch["label"].to(device)
                if cfg.get("channels_last", False):
                    if dataset.is_3d and hasattr(torch, "channels_last_3d"):
                        inputs = inputs.to(memory_format=torch.channels_last_3d)
                        labels = labels.to(memory_format=torch.channels_last_3d)
                    else:
                        inputs = inputs.to(memory_format=torch.channels_last)
                        labels = labels.to(memory_format=torch.channels_last)

                with autocast_context():
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                val_loss += float(loss.item())
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss = val_loss / max(val_steps, 1)
        epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            tqdm.write(f"Checkpoint saved: {model_save_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                tqdm.write(f"Early stopping triggered (patience={patience}).")
                break

if __name__ == "__main__":
    main()