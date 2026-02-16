import os
import sys
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

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _pin_to_single_mig_slice_for_rank() -> None:
    """Pin each torchrun worker to exactly one MIG slice.

    Why: some libraries default to cuda:0. If every rank can see all slices,
    they may all pile onto the first one.

    Mechanism: SLURM exports MIG_UUIDS_CSV as a comma-separated list of MIG UUIDs
    that were allocated for this job. We map LOCAL_RANK -> that UUID and then
    overwrite CUDA_VISIBLE_DEVICES to only that UUID.
    """

    mig_csv = os.environ.get("MIG_UUIDS_CSV")
    local_rank_str = os.environ.get("LOCAL_RANK")
    if not mig_csv or local_rank_str is None:
        return

    mig_uuids = [u.strip() for u in mig_csv.split(",") if u.strip()]
    if not mig_uuids:
        return

    try:
        local_rank = int(local_rank_str)
    except ValueError:
        return

    if 0 <= local_rank < len(mig_uuids):
        os.environ["CUDA_VISIBLE_DEVICES"] = mig_uuids[local_rank]


_pin_to_single_mig_slice_for_rank()

import torch
import torch.distributed as dist


def setup_distributed(backend="gloo"):
    """Initialize distributed process group if launched via torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        return rank, world_size
    else:
        return 0, 1


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


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


def process_all_patients(input_root, output_root, anatomy, rank=0, world_size=1):
    classes = cfg["ANATOMY_CLASSES"][anatomy]
    input_path = Path(input_root)
    output_path = Path(output_root)

    # Collect all valid patient directories
    patient_dirs = sorted([p for p in input_path.iterdir() if p.is_dir()])

    # Partition patients across ranks (round-robin for balanced load)
    my_patients = patient_dirs[rank::world_size]

    pbar = tqdm(
        my_patients,
        desc=f"Rank {rank}",
        unit="patient",
        position=rank,
        leave=True,
        bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    for patient_dir in pbar:
        patient_id = patient_dir.name
        pbar.set_postfix_str(patient_id)

        ct_file = patient_dir / "ct.nii.gz"
        mask_file = patient_dir / "mask.nii.gz"

        if not ct_file.exists() or not mask_file.exists():
            print(f"Rank {rank} WARNING: Missing ct.nii.gz or mask.nii.gz for patient {patient_id}. Skipping.")
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

    # Synchronize all workers before exiting
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed TotalSegmentator segmentation across MIG slices.")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Path to the input folder containing patient subfolders.")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the output folder where processed segmentations will be saved.")
    parser.add_argument("--anatomy", required=True, choices=["AB", "TH", "HN"], 
                        help="Anatomy type (AB, TH, HN).", default="AB")
    parser.add_argument("--dist_backend", type=str, default="gloo",
                        choices=["gloo", "nccl"],
                        help="Distributed backend (default: gloo, recommended for MIG).")
    args = parser.parse_args()

    rank, world_size = setup_distributed(backend=args.dist_backend)

    try:
        process_all_patients(args.input_folder, args.output_folder, args.anatomy, 
                           rank=rank, world_size=world_size)
    finally:
        cleanup_distributed()