from pathlib import Path

from src.io.load import load_case
from src.slicing.selection import select_axial_slices
from src.slicing.extract import extract_axial_slices
from src.panels.grid import make_grid
from src.panels.diff import compute_diff_slices
from src.io.save import save_image

from src.vlm.stage1 import Stage1VLM
from src.vlm.stage2 import Stage2VLM
from src.vlm.prompts import stage1_prompt, stage2_prompt

from src.metrics.loader import load_metrics, get_case_metrics


# -------- CONFIG --------
DATA_DIR = Path("data")
OUTPUT_PANEL_DIR = Path("outputs/panels")
OUTPUT_STAGE1_DIR = Path("outputs/stage1")
OUTPUT_STAGE2_DIR = Path("outputs/stage2")

rows, cols = 3, 4
size = 128


# -------- SETUP --------
OUTPUT_PANEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_STAGE1_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_STAGE2_DIR.mkdir(parents=True, exist_ok=True)

cases = [p.name for p in DATA_DIR.iterdir() if p.is_dir()]

df = load_metrics(DATA_DIR / "per_patient_metrics.csv")

stage1_model = Stage1VLM()
stage2_model = Stage2VLM()


# -------- LOOP --------
for case_id in cases:
    print(f"\nProcessing {case_id}...")

    case_path = DATA_DIR / case_id

    # ---- Load ----
    case = load_case(case_path)

    # ---- Slice ----
    indices = select_axial_slices(case["ct"], num_slices=12)
    slices = extract_axial_slices(case, indices)

    # ---- Diff ----
    diff_slices = compute_diff_slices(slices["ct"], slices["sct"])

    # ---- Grids ----
    ct_grid = make_grid(slices["ct"], rows, cols, size)
    sct_grid = make_grid(slices["sct"], rows, cols, size)
    mri_grid = make_grid(slices["mri"], rows, cols, size)
    diff_grid = make_grid(diff_slices, rows, cols, size)

    import numpy as np
    combined = np.concatenate([ct_grid, sct_grid, mri_grid, diff_grid], axis=1)

    panel_path = OUTPUT_PANEL_DIR / f"{case_id}.png"
    save_image(combined, panel_path)

    # ---- Stage 1 ----
    prompt1 = stage1_prompt()
    stage1_output = stage1_model.generate(panel_path, prompt1)

    stage1_path = OUTPUT_STAGE1_DIR / f"{case_id}.txt"
    stage1_path.write_text(stage1_output)

    # ---- Metrics ----
    metrics = get_case_metrics(df, case_id)

    # ---- Stage 2 ----
    prompt2 = stage2_prompt(stage1_output, metrics)
    stage2_output = stage2_model.generate(prompt2)

    stage2_path = OUTPUT_STAGE2_DIR / f"{case_id}.txt"
    stage2_path.write_text(stage2_output)

    print(f"Done {case_id}")
