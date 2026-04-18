"""Run ablation inference pipelines across multiple TorchScript checkpoints.

How this file runs
------------------
For each model checkpoint in MODEL_PATHS, this script:
1) Selects inference mode (2D or 3D) and matching patch configuration.
2) Runs inference/model_test.py to generate per-patient sct.mha predictions.
3) Runs inference/postproc.py to denormalize and mask predictions into
   sct_postproc.mha.
4) Runs inference/calc_metrics.py to compute aggregate image metrics.
5) Appends one row to a CSV summary with run metadata and metric values.
6) Deletes per-model generated inference files by default (temporary outputs),
   unless --keep-run-files is provided.

How to run
----------
From project root:
    python ablation_inferences.py

Common variants:
    python ablation_inferences.py --device cpu
    python ablation_inferences.py --stop-on-error
    python ablation_inferences.py --keep-run-files
    python ablation_inferences.py --outputs-root inference/ablation_outputs_hn

Output behavior
---------------
- Summary CSV is saved to:
    <outputs-root>/ablation_summary.csv
- Per-model folders are created under <outputs-root>/<run_name> during execution.
- Those per-model folders are removed after each run unless --keep-run-files is set.

Available argparse options
--------------------------
--root PATH
    Project root folder. Default: directory containing this file.

--input-path PATH
    Dataset root containing patient folders. Default is the current HN dataset path
    configured in this script.

--modality {mr,cbct}
    Input modality expected by inference scripts. Default: mr.

--device {cpu,cuda}
    Device used by model_test.py. Default: cuda.

--channels-last / --no-channels-last
    Enable or disable channels_last memory format during inference.
    Default: enabled.

--use-preproc-files / --no-preproc-files
    Read <modality>_preproc.mha when enabled, otherwise <modality>.mha.
    Default: enabled.

--outputs-root PATH
    Root folder where per-model outputs and ablation_summary.csv are written.
    Default: inference/ablation_outputs.

--stop-on-error
    Stop remaining model runs immediately after first failure.

--keep-run-files
    Keep generated per-model inference output directories instead of deleting
    them after metrics are collected.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

# Detect interactive environment (Jupyter/IPython)
try:
    get_ipython()  # type: ignore
    IS_INTERACTIVE = True
except NameError:
    IS_INTERACTIVE = False

# Conditional tqdm import
if IS_INTERACTIVE:
    try:
        from tqdm.auto import tqdm as tqdm_iter
    except ImportError:
        tqdm_iter = lambda x, **kwargs: x  # type: ignore
else:
    tqdm_iter = lambda x, **kwargs: x  # type: ignore


SliceMode = Union[int, str]


@dataclass(frozen=True)
class ModelSpec:
    model_path: str
    patch_size: Tuple[int, ...]
    num_adjacent_slices: SliceMode


# Model list
MODEL_PATHS = [
    # 2D
    # "ablations/Unet2D/HN/preproc_l1_ssim/run2/checkpoints/scripted.pth",
    # "ablations/Unet2D/HN/preproc_l1_ssim_dice/run1/checkpoints/scripted.pth",
    # "ablations/Unet2D/HN/pruned_preproc_l1_ssim/checkpoints/scripted.pth",
    # "ablations/Unet2D/HN/pruned_preproc_l1_ssim_dice/run1/checkpoints/scripted.pth",
    # 3D
    # "ablations/Unet3D/HN/preproc_l1_ssim/run2/checkpoints/scripted.pth",
    # "ablations/Unet3D/HN/preproc_l1_ssim_dice/run1/checkpoints/scripted.pth",
    # "ablations/Unet3D/HN/pruned_preproc_l1_ssim/run2/checkpoints/scripted.pth",
    # "ablations/Unet3D/HN/pruned_preproc_l1_ssim_dice/run1/checkpoints/scripted.pth",
    "ablations/Unet3D/HN/pruned_preproc_l1_ssim_dice/run1/epoch_78_scripted.pt"
]


def build_model_specs() -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for idx, model_path in enumerate(MODEL_PATHS):
        # is_3d = idx >= 4
        is_3d = "Unet3D" in model_path
        if is_3d:
            specs.append(ModelSpec(model_path=model_path, patch_size=(32, 192, 192), num_adjacent_slices="3d"))
        else:
            specs.append(ModelSpec(model_path=model_path, patch_size=(192, 192), num_adjacent_slices=0))
    return specs


def to_abs(root: Path, maybe_rel: str) -> Path:
    p = Path(maybe_rel)
    return p if p.is_absolute() else root / p


def make_run_name(model_path: Path) -> str:
    parts = list(model_path.parts)
    if "ablations" in parts:
        i = parts.index("ablations")
        suffix = parts[i + 1 : -2]  # drop checkpoints/scripted.pth
        return "__".join(suffix)
    return model_path.stem


def run_cmd(cmd: List[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_cmd_capture(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


def run_model_test(
    inference_dir: Path,
    model_path: Path,
    input_path: Path,
    output_dir: Path,
    modality: str,
    patch_size: Tuple[int, ...],
    num_adjacent_slices: SliceMode,
    device: str,
    channels_last: bool,
    use_preproc_files: bool,
) -> float:
    print("  [Stage 1/3] Running model inference (model_test.py)...")
    snippet = (
        "import argparse; "
        "import config; "
        "import model_test; "
        f"config.MODEL_PATH={model_path.as_posix()!r}; "
        f"config.INPUT_PATH={input_path.as_posix()!r}; "
        f"config.OUTPUT_DIR={output_dir.as_posix()!r}; "
        f"config.MODALITY={modality!r}; "
        f"config.PATCH_SIZE={tuple(patch_size)!r}; "
        f"config.NUM_ADJACENT_SLICES={num_adjacent_slices!r}; "
        f"config.DEVICE={device!r}; "
        f"config.CHANNELS_LAST={channels_last!r}; "
        f"config.USE_PREPROC_FILES={use_preproc_files!r}; "
        "model_test.main(argparse.Namespace(quiet=True))"
    )
    start = time.perf_counter()
    run_cmd([sys.executable, "-c", snippet], cwd=inference_dir)
    elapsed = time.perf_counter() - start
    print(f"  [Stage 1/3] Model inference completed in {elapsed:.2f}s.")
    return elapsed


def run_postproc(inference_dir: Path, input_path: Path, output_dir: Path, modality: str) -> None:
    print("  [Stage 2/3] Running postprocessing (postproc.py)...")
    cmd = [
        sys.executable,
        "postproc.py",
        "--org_ct",
        str(input_path),
        "--sct",
        str(output_dir),
        "--sct_name",
        "sct.mha",
        "--out_name",
        "sct_postproc.mha",
        "--modality",
        modality,
    ]
    run_cmd(cmd, cwd=inference_dir)
    print("  [Stage 2/3] Postprocessing completed.")


def run_calc_metrics(
    inference_dir: Path,
    input_path: Path,
    output_dir: Path,
    modality: str,
) -> Dict[str, float]:
    print("  [Stage 3/3] Computing metrics (calc_metrics.py)...")
    snippet = (
        "import sys, types; import config; "
        f"config.INPUT_PATH={input_path.as_posix()!r}; "
        f"config.OUTPUT_DIR={output_dir.as_posix()!r}; "
        f"config.MODALITY={modality!r}; "
        "sys.modules.setdefault('inference', types.ModuleType('inference')).config = config; "
        "sys.modules['inference.config'] = config; "
        "import calc_metrics; "
        "calc_metrics.main()"
    )
    proc = run_cmd_capture([sys.executable, "-c", snippet], cwd=inference_dir)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    print("  [Stage 3/3] Metrics computation completed.")
    return parse_metrics_from_stdout(proc.stdout)


def parse_metrics_from_stdout(stdout: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    pattern = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$")
    for line in stdout.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        key = match.group(1)
        metrics[key] = float(match.group(2))
    if not metrics:
        raise ValueError("Could not parse metrics from calc_metrics output")
    return metrics


def cleanup_run_output(run_out_dir: Path) -> None:
    if run_out_dir.exists():
        shutil.rmtree(run_out_dir)


def write_summary_csv(path: Path, rows: List[Dict[str, object]], metric_keys: Set[str]) -> None:
    fields = ["run_name", "model_path", "inference_time_gpu", "status", "error"] + sorted(metric_keys)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation inference pipelines")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root folder",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path(
            "/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/datasets/bhuvaneshsingla/synthrad2025-hn/versions/3/Task1"
        ),
        help="Dataset root with patient folders",
    )
    parser.add_argument("--modality", default="mr", choices=["mr", "cbct"])
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--channels-last", action="store_true", default=True)
    parser.add_argument("--no-channels-last", action="store_false", dest="channels_last")
    parser.add_argument("--use-preproc-files", action="store_true", default=True)
    parser.add_argument("--no-preproc-files", action="store_false", dest="use_preproc_files")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("inference/ablation_outputs"),
        help="Folder under root where per-model outputs are written",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop all runs immediately when one model fails",
    )
    parser.add_argument(
        "--keep-run-files",
        action="store_true",
        help="Keep per-model inference files instead of deleting them after metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    inference_dir = root / "inference"
    outputs_root = to_abs(root, str(args.outputs_root)).resolve()
    outputs_root.mkdir(parents=True, exist_ok=True)

    specs = build_model_specs()
    failures: List[str] = []
    summary_rows: List[Dict[str, object]] = []
    metric_keys: Set[str] = set()

    print("\n" + "="*80)
    print("ABLATION INFERENCE PIPELINE")
    print("="*80)
    print(f"Total models to process: {len(specs)}")
    print(f"Dataset path: {args.input_path}")
    print(f"Output root: {outputs_root}")
    print(f"Device: {args.device}")
    print(f"Modality: {args.modality}")
    print(f"Keep files: {args.keep_run_files}")
    print("="*80 + "\n")

    for idx, spec in enumerate(tqdm_iter(specs, desc="Model Progress", disable=not IS_INTERACTIVE), start=1):
        model_abs = to_abs(root, spec.model_path).resolve()
        run_name = make_run_name(model_abs)
        run_out_dir = outputs_root / run_name
        run_out_dir.mkdir(parents=True, exist_ok=True)
        row: Dict[str, object] = {
            "run_name": run_name,
            "model_path": str(model_abs),
            "inference_time_gpu": "",
            "status": "failed",
            "error": "",
        }

        print(f"\n[{idx}/{len(specs)}] ===== Running: {run_name} =====")
        print(f"  Model path: {model_abs}")
        print(f"  Output dir: {run_out_dir}")
        print(f"  Patch size: {spec.patch_size}")
        print(f"  Adjacent slices: {spec.num_adjacent_slices}")

        if not model_abs.exists():
            msg = f"Missing model checkpoint: {model_abs}"
            print(f"  [ERROR] {msg}")
            failures.append(msg)
            row["error"] = msg
            summary_rows.append(row)
            if args.stop_on_error:
                break
            continue

        try:
            inference_time_gpu = run_model_test(
                inference_dir=inference_dir,
                model_path=model_abs,
                input_path=args.input_path.resolve(),
                output_dir=run_out_dir,
                modality=args.modality,
                patch_size=spec.patch_size,
                num_adjacent_slices=spec.num_adjacent_slices,
                device=args.device,
                channels_last=args.channels_last,
                use_preproc_files=args.use_preproc_files,
            )
            row["inference_time_gpu"] = round(inference_time_gpu, 3)
            run_postproc(inference_dir=inference_dir, input_path=args.input_path.resolve(), output_dir=run_out_dir, modality=args.modality)
            metrics = run_calc_metrics(
                inference_dir=inference_dir,
                input_path=args.input_path.resolve(),
                output_dir=run_out_dir,
                modality=args.modality,
            )
            row.update(metrics)
            metric_keys.update(metrics.keys())
            row["status"] = "ok"
        except subprocess.CalledProcessError as exc:
            msg = f"Run failed for {run_name} (exit={exc.returncode})"
            print(f"  [ERROR] {msg}")
            if exc.stdout:
                print(f"  [STDOUT] {exc.stdout}")
            if exc.stderr:
                print(f"  [STDERR] {exc.stderr}")
            failures.append(msg)
            row["error"] = msg
            summary_rows.append(row)
            if args.stop_on_error:
                break
            if not args.keep_run_files:
                cleanup_run_output(run_out_dir)
            continue
        except ValueError as exc:
            msg = f"Metrics parse failed for {run_name}: {exc}"
            print(f"  [ERROR] {msg}")
            failures.append(msg)
            row["error"] = msg
            summary_rows.append(row)
            if args.stop_on_error:
                break
            if not args.keep_run_files:
                cleanup_run_output(run_out_dir)
            continue

        summary_rows.append(row)
        print(f"  [SUCCESS] Model run completed and metrics recorded.")
        if not args.keep_run_files:
            print(f"  [INFO] Cleaning up temporary output files...")
            cleanup_run_output(run_out_dir)

    print("\n" + "="*80)
    print("WRITING SUMMARY CSV")
    print("="*80)
    summary_path = outputs_root / "ablation_summary.csv"
    write_summary_csv(summary_path, summary_rows, metric_keys)
    print(f"Summary CSV saved: {summary_path}")
    print(f"Total models processed: {len(specs)}")
    print(f"Successful runs: {sum(1 for r in summary_rows if r['status'] == 'ok')}")
    print(f"Failed runs: {sum(1 for r in summary_rows if r['status'] == 'failed')}")
    print("="*80)

    if failures:
        print("\n[FINAL REPORT] Completed with failures:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)

    print("\n[FINAL REPORT] All ablation inference runs completed successfully ✓")


if __name__ == "__main__":
    main()

