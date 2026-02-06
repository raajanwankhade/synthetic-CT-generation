import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


def _seed_worker(worker_id: int, seed: int):
    worker_seed = (seed + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(dataset: Dataset, cfg, seed: int, device: torch.device):
    val_split = cfg.data.val_split

    patient_ids = np.array([str(p) for (p, _) in dataset.samples])
    uniq_patients = np.unique(patient_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq_patients)

    n_val_patients = int(len(uniq_patients) * val_split)
    val_patients = set(uniq_patients[:n_val_patients])
    train_patients = set(uniq_patients[n_val_patients:])

    train_idx = np.where(np.isin(patient_ids, list(train_patients)))[0].tolist()
    val_idx = np.where(np.isin(patient_ids, list(val_patients)))[0].tolist()

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    g = torch.Generator()
    g.manual_seed(seed)

    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers_train,
        pin_memory=pin,
        persistent_workers=(cfg.dataloader.num_workers_train > 0),
        worker_init_fn=lambda wid: _seed_worker(wid, seed),
        generator=g,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers_val,
        pin_memory=pin,
        persistent_workers=(cfg.dataloader.num_workers_val > 0),
        worker_init_fn=lambda wid: _seed_worker(wid, seed),
        generator=g,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )

    return train_loader, val_loader, train_ds, val_ds, train_patients, val_patients
