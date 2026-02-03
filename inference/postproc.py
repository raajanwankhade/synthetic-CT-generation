#!/usr/bin/env python3
import os
import sys
import argparse
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
    p.add_argument("--sct", required=True,
                   help="Root folder of sCT patient folders")
    p.add_argument("--sct_name", default="sct.mha",
                   help="sCT filename inside each patient folder (default: sct.mha)")
    p.add_argument("--out_name", default="sct_postproc.mha")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    root = args.org_ct

    stats_dir = os.path.join(root, STATS_DIRNAME)
    if not (os.path.exists(os.path.join(stats_dir, MEAN_FILENAME)) and
            os.path.exists(os.path.join(stats_dir, STD_FILENAME))):
        print("ERROR: stats files not found. Run preprocessing first.")
        sys.exit(1)

    mean = float(np.load(os.path.join(stats_dir, MEAN_FILENAME)))
    std = float(np.load(os.path.join(stats_dir, STD_FILENAME)))
    out_name = args.out_name
    postprocess_sct_to_hu(root, args.sct, mean, std,
                          sct_name=args.sct_name,
                          out_name=out_name)
    print("[DONE] Postprocessing complete.")

if __name__ == "__main__":
    main()
