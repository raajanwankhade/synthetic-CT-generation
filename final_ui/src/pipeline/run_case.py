from pathlib import Path
import numpy as np

from src.io.load import load_case
from src.slicing.selection import select_axial_slices
from src.slicing.extract import extract_axial_slices
from src.panels.grid import make_grid
from src.panels.diff import compute_diff_slices
from src.io.save import save_image

from src.vlm.prompts import stage1_prompt, stage2_prompt
from src.metrics.loader import load_metrics, get_case_metrics


def run_pipeline_for_case(case_id, case_path, stage1_model, stage2_model, output_root="outputs"):
    output_root = Path(output_root)

    panel_dir = output_root / "panels"
    stage1_dir = output_root / "stage1"
    stage2_dir = output_root / "stage2"

    panel_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load ----
    case = load_case(case_path)

    # ---- Slice ----
    indices = select_axial_slices(case["ct"], num_slices=12)
    slices = extract_axial_slices(case, indices)

    # ---- Diff ----
    diff_slices = compute_diff_slices(slices["ct"], slices["sct"])

    # ---- Grids ----
    rows, cols, size = 3, 4, 128

    ct_grid = make_grid(slices["ct"], rows, cols, size)
    sct_grid = make_grid(slices["sct"], rows, cols, size)
    mri_grid = make_grid(slices["mri"], rows, cols, size)
    diff_grid = make_grid(diff_slices, rows, cols, size)

    combined = np.concatenate([ct_grid, sct_grid, mri_grid, diff_grid], axis=1)

    panel_path = panel_dir / f"{case_id}.png"
    save_image(combined, panel_path)

    # ---- Stage 1 ----
    s1 = stage1_model.generate(panel_path, stage1_prompt())
    (stage1_dir / f"{case_id}.txt").write_text(s1)

    # ---- Metrics ----
    df = load_metrics("data/per_patient_metrics.csv")
    metrics = get_case_metrics(df, case_id)

    # ---- Stage 2 ----
    s2 = stage2_model.generate(stage2_prompt(s1, metrics))
    (stage2_dir / f"{case_id}.txt").write_text(s2)

    return panel_path