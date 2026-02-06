import os
import argparse
from data.normalization import (
    preprocess_cbct_and_save,
    preprocess_mri_and_save,
    preprocess_ct_and_save,
)
# ---------------- CLI ----------------
def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--root_folder", required=True,
                   help="Root folder with patient subfolders")
    p.add_argument("--modality", required=True, choices=["mri", "cbct"],
                   help="Select preprocessing modality: mri or cbct")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    root = args.root_folder
    modality = args.modality.lower()

    ct_mean, ct_std = preprocess_ct_and_save(root)
    print(f"[DONE] Preprocessing CT complete. mean={ct_mean:.2f}, std={ct_std:.2f}")

    if modality == "cbct":
        cbct_mean, cbct_std = preprocess_cbct_and_save(root)
        print(f"[DONE] Preprocessing CBCT complete. mean={cbct_mean:.2f}, std={cbct_std:.2f}")
   
    elif modality == "mri":
        preprocess_mri_and_save(root)
        print("[DONE] Preprocessing MRI complete.")

if __name__ == "__main__":
    main()
