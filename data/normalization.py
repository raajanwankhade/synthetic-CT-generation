import json
import os
from glob import glob

import numpy as np
from tqdm import tqdm

from utils.constants import (
    CLIP_MIN,
    CLIP_MAX,
    STATS_DIRNAME,
    MEAN_FILENAME,
    STD_FILENAME,
    METADATA_FILENAME,
    MASK_FILENAME,
    CT_FILENAME,
    CT_PREPROC_FILENAME,
    CBCT_FILENAME,
    CBCT_PREPROC_FILENAME,
    MR_FILENAME,
    MR_PREPROC_FILENAME,
)
from utils.io import load_image_as_array, write_array_as_image


class RunningStats:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def push(self, x):
        if x.size == 0:
            return
        x = x.astype(np.float64)
        n = x.size
        new_mean = x.mean()
        new_var = x.var(ddof=1) if n > 1 else 0.0

        delta = new_mean - self.mean
        total = self.count + n

        self.M2 += new_var * (n - 1) + delta**2 * self.count * n / total
        self.mean += delta * n / total
        self.count = total

    def finalize(self):
        if self.count < 2:
            return float(self.mean), 0.0
        var = self.M2 / (self.count - 1)
        return float(self.mean), float(np.sqrt(var))


def compute_population_stats(root_dir, in_name=CT_FILENAME):
    rs = RunningStats()
    patient_dirs = sorted([d for d in glob(os.path.join(root_dir, "*")) if os.path.isdir(d)])

    for pd in tqdm(patient_dirs, desc="Computing population stats"):
        ct_path = os.path.join(pd, in_name)
        mask_path = os.path.join(pd, MASK_FILENAME)
        if not (os.path.exists(ct_path) and os.path.exists(mask_path)):
            continue

        _, ct = load_image_as_array(ct_path)
        _, mask = load_image_as_array(mask_path)

        mask = mask > 0
        ct = np.clip(ct, CLIP_MIN, CLIP_MAX).astype(np.float32)

        rs.push(ct[mask])

    mean, std = rs.finalize()

    stats_dir = os.path.join(root_dir, STATS_DIRNAME)
    os.makedirs(stats_dir, exist_ok=True)
    np.save(os.path.join(stats_dir, MEAN_FILENAME), mean)
    np.save(os.path.join(stats_dir, STD_FILENAME), std)

    meta = {
        "clip_min": CLIP_MIN,
        "clip_max": CLIP_MAX,
        "patients_used": len(patient_dirs),
    }

    with open(os.path.join(stats_dir, METADATA_FILENAME), "w") as f:
        json.dump(meta, f, indent=2)

    return mean, std


def apply_population_normalization_and_save(
    root_dir,
    mean,
    std,
    in_name=CT_FILENAME,
    out_name=CT_PREPROC_FILENAME,
):
    patient_dirs = sorted([d for d in glob(os.path.join(root_dir, "*")) if os.path.isdir(d)])

    for pd in tqdm(patient_dirs, desc="Saving normalized CT"):
        ct_path = os.path.join(pd, in_name)
        mask_path = os.path.join(pd, MASK_FILENAME)
        if not (os.path.exists(ct_path) and os.path.exists(mask_path)):
            continue

        ref_img, ct = load_image_as_array(ct_path)
        _, mask = load_image_as_array(mask_path)

        mask = mask > 0
        ct = np.clip(ct, CLIP_MIN, CLIP_MAX).astype(np.float32)
        ct[~mask] = CLIP_MIN

        if std == 0:
            std = 1e-6

        z = ((ct - mean) / std).astype(np.float32)
        write_array_as_image(z, ref_img, os.path.join(pd, out_name), use_compression=True)


def preprocess_cbct_and_save(
    root_dir,
    in_name=CBCT_FILENAME,
    out_name=CBCT_PREPROC_FILENAME,
):
    mean, std = compute_population_stats(root_dir, in_name=in_name)
    patient_dirs = sorted([d for d in glob(os.path.join(root_dir, "*")) if os.path.isdir(d)])

    for pd in tqdm(patient_dirs, desc="Saving normalized CBCT"):
        cbct_path = os.path.join(pd, in_name)
        mask_path = os.path.join(pd, MASK_FILENAME)
        if not (os.path.exists(cbct_path) and os.path.exists(mask_path)):
            continue

        ref_img, cbct = load_image_as_array(cbct_path)
        _, mask = load_image_as_array(mask_path)

        mask = mask > 0
        cbct = np.clip(cbct, CLIP_MIN, CLIP_MAX).astype(np.float32)
        cbct[~mask] = CLIP_MIN

        if std == 0:
            std = 1e-6

        z = ((cbct - mean) / std).astype(np.float32)
        write_array_as_image(z, ref_img, os.path.join(pd, out_name), use_compression=True)

    return mean, std


def preprocess_mri_and_save(
    root_dir,
    in_name=MR_FILENAME,
    out_name=MR_PREPROC_FILENAME,
):
    patient_dirs = sorted([d for d in glob(os.path.join(root_dir, "*")) if os.path.isdir(d)])

    for pd in tqdm(patient_dirs, desc="Saving normalized MR"):
        mri_path = os.path.join(pd, in_name)
        if not os.path.exists(mri_path):
            continue

        ref_img, mri = load_image_as_array(mri_path)
        mri = mri.astype(np.float32)

        mean = mri.mean()
        std = mri.std()
        if std == 0:
            std = 1e-6

        z = ((mri - mean) / std).astype(np.float32)
        write_array_as_image(z, ref_img, os.path.join(pd, out_name), use_compression=True)


def preprocess_ct_and_save(root, in_name=CT_FILENAME, out_name=CT_PREPROC_FILENAME):
    mean, std = compute_population_stats(root, in_name=in_name)
    apply_population_normalization_and_save(root, mean, std, in_name=in_name, out_name=out_name)
    return mean, std
