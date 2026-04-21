import streamlit as st
from pathlib import Path
from PIL import Image

from src.metrics.loader import load_metrics, get_case_metrics
from src.postprocess.parser import parse_stage2_output
from src.postprocess.image_grid import visualize_panel

import numpy as np

# ---------- CONFIG ----------
DATA_DIR = Path("data")
STAGE2_DIR = Path("outputs/stage2")
PANEL_DIR = Path("outputs/panels")


# ---------- UI SETUP ----------
st.set_page_config(layout="wide")
st.title("sCT Interpretability Dashboard")


# ---------- CASE LIST ----------
cases = [p.name for p in DATA_DIR.iterdir() if p.is_dir()]

if not cases:
    st.error("No cases found in data directory.")
    st.stop()

case_id = st.selectbox("Select Case", cases)


# ---------- LOAD METRICS ----------
metrics = None
try:
    df = load_metrics(DATA_DIR / "per_patient_metrics.csv")
    metrics = get_case_metrics(df, case_id)
except Exception as e:
    st.error(f"Error loading metrics: {e}")


# ---------- LOAD IMAGE ----------
image = None
image_path = PANEL_DIR / f"{case_id}.png"

if image_path.exists():
    try:
        image = Image.open(image_path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
else:
    st.warning(f"Panel not found for case: {case_id}")


# ---------- LOAD STAGE 2 ----------
parsed = None
stage2_path = STAGE2_DIR / f"{case_id}.txt"

if stage2_path.exists():
    try:
        stage2_text = stage2_path.read_text()
        parsed = parse_stage2_output(stage2_text)
    except Exception as e:
        st.error(f"Error parsing Stage 2 output: {e}")
else:
    st.warning(f"Stage 2 output not found for case: {case_id}")


# ---------- IMAGE (FULL WIDTH) ----------
st.subheader("CT | sCT | MRI | DIFF")

if image is not None:
    # st.image(image, use_container_width=True)
    panel_np = np.array(image)
    panel_vis = visualize_panel(panel_np)

    st.image(panel_vis, use_container_width=True)
else:
    st.info("No image available for this case.")


st.subheader("Metrics")
if metrics:
    st.write(f"**MAE:** {metrics['mae']}")
    st.write(f"**PSNR:** {metrics['psnr']}")
    st.write(f"**MS-SSIM:** {metrics['ms_ssim']}")

st.divider()


# ---------- TEXT SECTION ----------
col1, col2 = st.columns(2)


# ----- LEFT: METRIC INTERPRETATION -----
with col1:
    st.subheader("Metric Interpretation")

    if parsed:
        st.markdown(f"**MAE:** {parsed['mae']}")
        st.markdown(f"**PSNR:** {parsed['psnr']}")
        st.markdown(f"**MS-SSIM:** {parsed['ms_ssim']}")
    else:
        st.info("Interpretation not available.")


# ----- RIGHT: SUMMARY -----
with col2:
    st.subheader("Summary")

    if parsed:
        st.write(parsed["summary"])
    else:
        st.info("Summary not available.")
# ---------- LAYOUT ----------
# # col1, col2 = st.columns([2, 1])


# # ---------- LEFT: IMAGE ----------
# with col1:
#     st.subheader("CT | sCT | MRI | DIFF")

#     if image is not None:
#         st.image(image, use_container_width=True)
#     else:
#         st.info("No image available for this case.")


# # ---------- RIGHT: METRICS + EXPLANATION ----------
# with col2:
#     st.subheader("Metrics")

#     if metrics:
#         st.write(f"**MAE:** {metrics['mae']}")
#         st.write(f"**PSNR:** {metrics['psnr']}")
#         st.write(f"**MS-SSIM:** {metrics['ms_ssim']}")
#     else:
#         st.info("Metrics not available.")

#     st.divider()

#     if parsed:
#         st.subheader("Interpretation")

#         st.markdown(f"**MAE:** {parsed['mae']}")
#         st.markdown(f"**PSNR:** {parsed['psnr']}")
#         st.markdown(f"**MS-SSIM:** {parsed['ms_ssim']}")

#         st.divider()

#         st.subheader("Summary")
#         st.write(parsed["summary"])
#     else:
#         st.info("Interpretation not available.")