from pathlib import Path
from typing import List
import argparse

import numpy as np
from tqdm.auto import tqdm

import pandas as pd

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

try:
    from inference import config
    from inference.image_metrics import ImageMetrics
    from inference.test_dataset_class import is_patient_dir, read_image_to_array
except ImportError:
    import config
    from image_metrics import ImageMetrics
    from test_dataset_class import is_patient_dir, read_image_to_array


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
    print("[calc_metrics.py] Starting metrics calculation...")
    args = parse_args()
    input_path = Path(config.INPUT_PATH)
    output_dir = Path(config.OUTPUT_DIR)

    print(f"[calc_metrics.py] Input path: {input_path}")
    print(f"[calc_metrics.py] Output dir: {output_dir}")
    print(f"[calc_metrics.py] GT CT filename: {args.gt_ct_filename}")
    print(f"[calc_metrics.py] Predicted CT filename: {args.pred_ct_filename}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if not output_dir.exists():
        raise FileNotFoundError(f"Output dir not found: {output_dir}")

    patients: List[Path] = []   
    if is_patient_dir(input_path, config.MODALITY):
        patients = [input_path]
    else:
        for p in sorted(input_path.iterdir()):
            if is_patient_dir(p, config.MODALITY):
                patients.append(p)

    if not patients:
        raise ValueError("No valid patient folders found")

    print(f"[calc_metrics.py] Found {len(patients)} patient(s) to score")
    img_metrics = ImageMetrics()
    # img_metrics = ImageMetricsGPU()
    per_patient = []

    for patient_dir in tqdm_iter(patients, desc="Scoring patients", disable=not IS_INTERACTIVE):
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
    print("[calc_metrics.py] Metrics calculation completed.\n")

    df = pd.DataFrame(per_patient)
    df.to_csv("per_patient_metrics.csv", index=False)
if __name__ == "__main__":
    main()
