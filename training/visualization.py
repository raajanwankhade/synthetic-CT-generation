import numpy as np
import torch
import matplotlib.pyplot as plt

from training.loss import LOSS_NAMES


@torch.no_grad()
def visualize_batch(model, dataloader, device=None, num_samples=4):
    """Visualize input/ground-truth/generated for a batch."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    batch = next(iter(dataloader))
    x = batch["input"].to(device)
    y = batch["label"].to(device)

    preds = model(x)

    x = x.cpu()
    y = y.cpu()
    preds = preds.cpu()

    n = min(num_samples, x.size(0))

    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(10, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        if x.dim() == 5:
            mid = x.shape[2] // 2
            inp = x[i, 0, mid].numpy()
            gt = y[i, 0, mid].numpy()
            gen = preds[i, 0, mid].numpy()
        else:
            inp = x[i, x.shape[1] // 2].numpy() if x.shape[1] > 1 else x[i, 0].numpy()
            gt = y[i, 0].numpy()
            gen = preds[i, 0].numpy()

        for j, img in enumerate([inp, gt, gen]):
            vmin, vmax = np.percentile(img, [2, 98])
            axes[i, j].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            axes[i, j].axis("off")

        axes[i, 0].set_title("Input")
        axes[i, 1].set_title("Ground Truth (CT)")
        axes[i, 2].set_title("Generated")

    plt.tight_layout()
    return fig


def plot_history(history: dict, save_path: str):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 18))

    # 1. Training & Validation Loss
    plt.subplot(4, 1, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Composite Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. SSIM
    plt.subplot(4, 1, 2)
    plt.plot(epochs, history["SSIM"], label="SSIM", color="tab:blue", linewidth=2)
    plt.title("SSIM over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. PSNR
    plt.subplot(4, 1, 3)
    plt.plot(epochs, history["PSNR"], label="PSNR", color="tab:orange", linewidth=2)
    plt.title("PSNR over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Dice Score
    plt.subplot(4, 1, 4)
    plt.plot(epochs, history["Dice_Score"], label="Dice Score", color="tab:green", linewidth=2)
    plt.title("Dice Score over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
