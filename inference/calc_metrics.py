from pathlib import Path
from typing import List
import argparse

import numpy as np
from tqdm.auto import tqdm

from config import cfg

from metrics.image import ImageMetrics

from data.utils import is_patient_dir, read_image_to_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate metrics for synthetic CT outputs.")
    parser.add_argument("--gt-ct-filename", default="ct.mha",
                        help="Ground-truth CT filename inside each patient folder (default: ct.mha)")
    parser.add_argument("--mask-filename", default="mask.mha",
                        help="Mask filename inside each patient folder (default: mask.mha)")
    parser.add_argument("--pred-ct-filename", default="sct_postproc.mha",
                        help="Predicted CT filename inside each output patient folder (default: sct_postproc.mha)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(cfg.inference.input_path)
    output_dir = Path(cfg.inference.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if not output_dir.exists():
        raise FileNotFoundError(f"Output dir not found: {output_dir}")

    patients: List[Path] = []   
    if is_patient_dir(input_path, cfg.inference.modality):
        patients = [input_path]
    else:
        for p in sorted(input_path.iterdir()):
            if is_patient_dir(p, cfg.inference.modality):
                patients.append(p)

    if not patients:
        raise ValueError("No valid patient folders found")

    img_metrics = ImageMetrics()
    per_patient = []

    for patient_dir in tqdm(patients, desc="Scoring patients"):
        gt_ct_path = patient_dir / args.gt_ct_filename
        mask_path = patient_dir / args.mask_filename
        pred_ct_path = output_dir / patient_dir.name / args.pred_ct_filename

        if not pred_ct_path.exists():
            print(f"Skipping {patient_dir.name}: missing {pred_ct_path}")
            continue

        gt_ct = read_image_to_array(gt_ct_path)
        pred_ct = read_image_to_array(pred_ct_path)
        mask = read_image_to_array(mask_path)

        scores = img_metrics.score_patient(gt_ct, pred_ct, mask)
        scores["patient"] = patient_dir.name
        per_patient.append(scores)

    if not per_patient:
        raise ValueError("No predictions found to score")

    metric_names = [k for k in per_patient[0] if k != "patient"]
    averages = {
        m: float(np.mean([p[m] for p in per_patient]))
        for m in metric_names
    }

    print("\n=== Average metrics over all patients ===")
    for k, v in averages.items():
        print(f"{k:10s}: {v:.4f}")

    # Uncomment for per-patient breakdown
    # print("\n=== Per-patient metrics ===")
    # for p in per_patient:
    #     print(p)


if __name__ == "__main__":
    main()
