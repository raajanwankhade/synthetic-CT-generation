"""Single-GPU TotalSegmentator segmentation generation.

Processes all patients sequentially on a single GPU — no distributed
training, no torchrun, no MIG pinning.
"""

import os
from typing import Any, cast

import nibabel as nib
from nibabel.loadsave import load as nib_load
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from totalsegmentator.python_api import totalsegmentator
import argparse
from tqdm import tqdm
from config import cfg

import torch


def create_grouped_mask_vectorized(ts_data, body_data, classes):
    """Renumber specified TS labels sequentially from 1 onwards."""
    # Start with zeros everywhere
    grouped_mask = np.zeros_like(ts_data, dtype=np.uint8)
    
    # Set fallback value for all body tissue
    fallback_value = len(classes) + 1
    grouped_mask[body_data > 0] = fallback_value
    
    # Create mapping: TS label -> new sequential number
    max_ts_label = max(classes)
    lookup = np.full(max_ts_label + 1, fallback_value, dtype=np.uint8)
    
    for new_label, ts_label in enumerate(classes, start=1):
        lookup[ts_label] = new_label
    
    # Apply renumbering (vectorized)
    valid_mask = (ts_data >= 0) & (ts_data <= max_ts_label)
    grouped_mask[valid_mask] = lookup[ts_data[valid_mask]]
    
    return grouped_mask


def process_all_patients(input_root: str, output_root: str, anatomy: str) -> None:
    """Run TotalSegmentator on every patient and save grouped segmentation masks."""

    classes = cfg["ANATOMY_CLASSES"][anatomy]
    input_path = Path(input_root)
    output_path = Path(output_root)

    # Collect all valid patient directories
    patient_dirs = sorted([p for p in input_path.iterdir() if p.is_dir()])

    # Skip already-processed patients
    remaining_dirs = []
    for p in patient_dirs:
        final_output = output_path / p.name / "segmentation.mha"
        if not final_output.exists():
            remaining_dirs.append(p)

    pbar = tqdm(
        remaining_dirs,
        desc="Processing",
        unit="patient",
        bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    for patient_dir in pbar:
        patient_id = patient_dir.name
        pbar.set_postfix_str(patient_id)

        ct_file = patient_dir / "ct.nii.gz"
        mask_file = patient_dir / "mask.nii.gz"

        if not ct_file.exists() or not mask_file.exists():
            continue

        # Create output patient directory
        patient_output_dir = output_path / patient_id
        patient_output_dir.mkdir(parents=True, exist_ok=True)
        final_output = patient_output_dir / "segmentation.mha"

        try:
            # 1. Run TotalSegmentator
            ts_result: Any = totalsegmentator(str(ct_file), task="total", ml=True, quiet=True)
            ts_img = ts_result[0] if isinstance(ts_result, tuple) else ts_result
            ts_img = cast(Any, ts_img)
            ts_data = np.asarray(ts_img.get_fdata(), dtype=np.int32)

            # 2. Load Binary Body Mask
            body_img = cast(Any, nib_load(str(mask_file)))
            body_data = np.asarray(body_img.get_fdata(), dtype=np.uint8)

            # 3. Create Grouped Mask (OPTIMIZED - vectorized)
            grouped_mask = create_grouped_mask_vectorized(ts_data, body_data, classes)

            # 4. Save to .mha
            final_sitk = sitk.GetImageFromArray(np.transpose(grouped_mask, (2, 1, 0)))
            final_sitk = sitk.Cast(final_sitk, sitk.sitkUInt8)

            # Use original CT as reference for spacing/direction/origin
            ref_sitk = sitk.ReadImage(str(ct_file))
            final_sitk.CopyInformation(ref_sitk)

            sitk.WriteImage(final_sitk, str(final_output), useCompression=True)

        except Exception as e:
            pbar.write(f"ERROR {patient_id}: {e}")

    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-GPU TotalSegmentator segmentation.")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Path to the input folder containing patient subfolders.")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the output folder where processed segmentations will be saved.")
    parser.add_argument("--anatomy", required=True, choices=["AB", "TH", "HN"], 
                        help="Anatomy type (AB, TH, HN).", default="AB")
    args = parser.parse_args()

    process_all_patients(args.input_folder, args.output_folder, args.anatomy)