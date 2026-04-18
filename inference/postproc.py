#!/usr/bin/env python3
import os
import sys
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

try:
    from inference.test_dataset_class import is_patient_dir
except ImportError:
    from test_dataset_class import is_patient_dir

# Detect interactive environment (Jupyter/IPython)
try:
    get_ipython()  # type: ignore
    IS_INTERACTIVE = True
except NameError:
    IS_INTERACTIVE = False

# Conditional tqdm - only show progress bar in interactive environments
if IS_INTERACTIVE:
    tqdm_iter = tqdm
else:
    tqdm_iter = lambda x, **kwargs: x  # type: ignore

# ---------------- Constants ----------------
CLIP_MIN = -1024
CLIP_MAX = 3000
STATS_DIRNAME = "stats"
MEAN_FILENAME = "mean.npy"
STD_FILENAME = "std.npy"

# ---------------- I/O helpers ----------------
def load_image_as_array(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return img, arr

def write_array_as_image(arr, ref_img, out_path):
    out_img = sitk.GetImageFromArray(arr)
    out_img.CopyInformation(ref_img)
    sitk.WriteImage(out_img, out_path)

# ---------------- Postprocessing ----------------
def postprocess_sct_to_hu(root_dir, sct_root, mean, std,
                          sct_name="sct.mha",
                          out_name="sct_postproc.mha",
                          modality="mr"):
    print(f"[postproc.py] Starting postprocessing")
    print(f"[postproc.py] Input CT root: {root_dir}")
    print(f"[postproc.py] sCT root: {sct_root}")
    patient_dirs = sorted([d for d in glob(os.path.join(root_dir, "*"))
                           if os.path.isdir(d) and is_patient_dir(Path(d), modality)])
    print(f"[postproc.py] Found {len(patient_dirs)} patients")
    print(f"[postproc.py] Starting to load and process {len(patient_dirs)} patients...")

    for pd in tqdm_iter(patient_dirs, desc="Postprocessing sCT", disable=not IS_INTERACTIVE):
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
    p.add_argument("--sct", required=True,
                   help="Root folder of sCT patient folders")
    p.add_argument("--sct_name", default="sct.mha",
                   help="sCT filename inside each patient folder (default: sct.mha)")
    p.add_argument("--out_name", default="sct_postproc.mha")
    p.add_argument("--modality", default="mr", choices=["mr", "cbct"],
                   help="Input modality (default: mr)")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    root = args.org_ct

    print("\n[postproc.py] Postprocessing pipeline started")
    print(f"[postproc.py] Organization CT root: {root}")
    stats_dir = os.path.join(root, STATS_DIRNAME)
    if not (os.path.exists(os.path.join(stats_dir, MEAN_FILENAME)) and
            os.path.exists(os.path.join(stats_dir, STD_FILENAME))):
        print(f"[postproc.py] ERROR: stats files not found in {stats_dir}")
        sys.exit(1)

    mean = float(np.load(os.path.join(stats_dir, MEAN_FILENAME)))
    std = float(np.load(os.path.join(stats_dir, STD_FILENAME)))
    print(f"[postproc.py] Loaded statistics - mean: {mean:.4f}, std: {std:.4f}")
    out_name = args.out_name
    postprocess_sct_to_hu(root, args.sct, mean, std,
                          sct_name=args.sct_name,
                          out_name=out_name,
                          modality=args.modality)
    print("[postproc.py] Postprocessing completed successfully.")

if __name__ == "__main__":
    main()
