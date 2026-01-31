#!/usr/bin/env python3
import os
import sys
import argparse
import json
from glob import glob

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# ---------------- Constants ----------------
CLIP_MIN = -1024
CLIP_MAX = 3000
STATS_DIRNAME = "stats"
MEAN_FILENAME = "mean.npy"
STD_FILENAME = "std.npy"
METADATA_FILENAME = "stats_meta.json"

# ---------------- I/O helpers ----------------
def load_image_as_array(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return img, arr

def write_array_as_image(arr, ref_img, out_path):
    out_img = sitk.GetImageFromArray(arr)
    out_img.CopyInformation(ref_img)
    sitk.WriteImage(out_img, out_path)

# ---------------- Running stats (vectorized) ----------------
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

# ---------------- Population stats ----------------
def compute_population_stats(root_dir):
    rs = RunningStats()
    patient_dirs = sorted([d for d in glob(os.path.join(root_dir, "*")) if os.path.isdir(d)])

    for pd in tqdm(patient_dirs, desc="Computing population stats"):
        ct_path = os.path.join(pd, "ct.mha")
        mask_path = os.path.join(pd, "mask.mha")
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

# ---------------- Apply normalization ----------------
def apply_population_normalization_and_save(root_dir, mean, std, out_name="ct_preproc.mha"):
    patient_dirs = sorted([d for d in glob(os.path.join(root_dir, "*")) if os.path.isdir(d)])

    for pd in tqdm(patient_dirs, desc="Saving normalized CT"):
        ct_path = os.path.join(pd, "ct.mha")
        mask_path = os.path.join(pd, "mask.mha")
        if not (os.path.exists(ct_path) and os.path.exists(mask_path)):
            continue

        ref_img, ct = load_image_as_array(ct_path)
        _, mask = load_image_as_array(mask_path)

        mask = mask > 0
        ct = np.clip(ct, CLIP_MIN, CLIP_MAX).astype(np.float32)
        ct[~mask] = CLIP_MIN

        if std == 0:
            std = 1e-6

        z = (ct - mean) / std
        write_array_as_image(z.astype(np.float32), ref_img,
                             os.path.join(pd, out_name))

# ---------------- Postprocessing ----------------
def postprocess_sct_to_hu(root_dir, sct_root, mean, std,
                          sct_name="sct.mha",
                          out_name="sct_postproc.mha"):
    patient_dirs = sorted([d for d in glob(os.path.join(root_dir, "*")) if os.path.isdir(d)])

    for pd in tqdm(patient_dirs, desc="Postprocessing sCT"):
        pat_name = os.path.basename(pd)
        sct_path = os.path.join(sct_root, pat_name, sct_name)
        mask_path = os.path.join(pd, "mask.mha")

        if not (os.path.exists(sct_path) and os.path.exists(mask_path)):
            continue

        ref_img, sct = load_image_as_array(sct_path)
        _, mask = load_image_as_array(mask_path)

        mask = mask > 0
        hu = sct.astype(np.float32) * std + mean
        hu[~mask] = CLIP_MIN
        hu = np.clip(hu, CLIP_MIN, CLIP_MAX)

        write_array_as_image(hu.astype(np.float32), ref_img,
                             os.path.join(sct_root, pat_name, out_name))

# ---------------- CLI ----------------
def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--org_ct", required=True,
                   help="Root folder with patient subfolders")
    p.add_argument("--mode", required=True, choices=("pre", "post"))
    p.add_argument("--sct",
                   help="Root folder of sCT patient folders (post only)")
    p.add_argument("--sct_name", default="sct.mha",
                   help="sCT filename inside each patient folder (default: sct.mha)")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    root = args.org_ct

    if args.mode == "pre":
        mean, std = compute_population_stats(root)
        apply_population_normalization_and_save(root, mean, std)
        print(f"[DONE] Preprocessing complete. mean={mean:.2f}, std={std:.2f}")

    else:
        if args.sct is None:
            print("ERROR: --sct is required in post mode")
            sys.exit(1)

        stats_dir = os.path.join(root, STATS_DIRNAME)
        mean = float(np.load(os.path.join(stats_dir, MEAN_FILENAME)))
        std = float(np.load(os.path.join(stats_dir, STD_FILENAME)))

        postprocess_sct_to_hu(root, args.sct, mean, std,
                              sct_name=args.sct_name,
                              out_name="sct_postproc.mha")
        print("[DONE] Postprocessing complete.")

if __name__ == "__main__":
    main()
