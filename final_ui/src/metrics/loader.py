import pandas as pd
from pathlib import Path


def load_metrics(csv_path):
    """
    Load the CSV once and return dataframe.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def get_case_metrics(df, patient_id):
    """
    Extract metrics for a given patient_id.

    Returns:
        dict with mae, psnr, ms_ssim
    """
    if "patient" not in df.columns:
        raise ValueError("CSV must contain 'patient' column")

    row = df[df["patient"] == patient_id]

    if row.empty:
        raise ValueError(f"No metrics found for patient: {patient_id}")

    row = row.iloc[0]

    required_cols = ["mae", "psnr", "ms_ssim"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    return {
        "mae": float(row["mae"]),
        "psnr": float(row["psnr"]),
        "ms_ssim": float(row["ms_ssim"]),
    }
