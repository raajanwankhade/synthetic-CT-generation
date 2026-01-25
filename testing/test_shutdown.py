import os
import json
secrets_path = '../secrets.json'
with open(secrets_path, 'r') as f:
    secrets = json.load(f)
os.environ["KAGGLE_USERNAME"] = secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = secrets["KAGGLE_KEY"]
os.environ["WANDB_API_KEY"] = secrets["wandb_API_KEY"]
import os
import time
from datetime import datetime
from typing import Optional

import wandb
import torch

# -----------------------------
# Config
# -----------------------------
TOTAL_EPOCHS = 50
SLEEP_PER_EPOCH = 4  # seconds
RUN_DIR = "run"
CKPT_NAME = "last_checkpoint.pth"
STOP_FLAG_PATH = os.environ.get("GRACEFUL_STOP_FLAG", os.path.join(RUN_DIR, "stop.flag"))

# W&B settings (can be overridden via env)
WANDB_PROJECT = "graceful-test"
WANDB_NAME = "os-shutdown-test"
WANDB_MODE = "online"

# -----------------------------
# Graceful shutdown handling
# -----------------------------
_stop_requested = False
_cleanup_done = False
_run_id: Optional[str] = None
_last_epoch: Optional[int] = None
_flag_ack_printed = False


def _cleanup(reason: str = "unknown", finish_wandb: bool = True):
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True

    epoch_to_save = 0 if _last_epoch is None else _last_epoch
    try:
        save_checkpoint(epoch_to_save, _run_id or "unknown", RUN_DIR, str("epoch") + CKPT_NAME)
        if finish_wandb:
            wandb.finish()
            print(f"[wandb] Finished run cleanly ({reason}).", flush=True)
    except Exception as e:
        print(f"[cleanup] Failed during {reason}: {e}", flush=True)


# -----------------------------
# Training simulation
# -----------------------------

def save_checkpoint(epoch: int, run_id: str, save_dir: str, filename: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, filename)
    payload = {
        "epoch": epoch,
        "total_epochs": TOTAL_EPOCHS,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(payload, ckpt_path)
    print(f"[checkpoint] Saved: {ckpt_path}", flush=True)
    return ckpt_path


def train_loop():
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_NAME,
        config={
            "total_epochs": TOTAL_EPOCHS,
            "sleep_per_epoch": SLEEP_PER_EPOCH,
        },
        mode=WANDB_MODE,
    )

    global _run_id, _last_epoch, _stop_requested, _flag_ack_printed
    _run_id = wandb.run.id if wandb.run else "unknown"
    print(f"[wandb] run_id={_run_id}", flush=True)
    print(f"[config] epochs={TOTAL_EPOCHS} sleep={SLEEP_PER_EPOCH}s run_dir={RUN_DIR} ckpt={CKPT_NAME}", flush=True)

    _last_epoch = None

    try:
        for epoch in range(1, TOTAL_EPOCHS + 1):
            # Simulate work in small chunks; do NOT shorten the epoch after SIGTERM
            remaining = SLEEP_PER_EPOCH
            step = 0.2
            while remaining > 0:
                if not _stop_requested and os.path.exists(STOP_FLAG_PATH):
                    _stop_requested = True
                    if not _flag_ack_printed:
                        _flag_ack_printed = True
                        print(
                            f"[train] Stop request flag detected: {STOP_FLAG_PATH}. Will stop after this epoch.",
                            flush=True,
                        )

                time.sleep(min(step, remaining))
                remaining -= step

            # Fake metrics
            loss = 1.0 / (epoch + 1)
            acc = epoch / TOTAL_EPOCHS

            wandb.log({"epoch": epoch, "loss": loss, "acc": acc})
            print(f"[train] epoch={epoch}/{TOTAL_EPOCHS} loss={loss:.4f} acc={acc:.3f}", flush=True)

            _last_epoch = epoch

            if _stop_requested:
                print("[train] Stop requested. Ending after this epoch.", flush=True)
                break

    finally:
        _cleanup(reason="finally", finish_wandb=True)


if __name__ == "__main__":
    train_loop()
