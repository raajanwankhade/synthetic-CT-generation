import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import os
import shutil

from src.metrics.loader import load_metrics, get_case_metrics
from src.postprocess.parser import parse_stage2_output
from src.postprocess.image_grid import visualize_panel
from src.pipeline.run_case import run_pipeline_for_case


# ---------- CONFIG ----------
DATA_DIR = Path("data")
STAGE2_DIR = Path("outputs/stage2")
PANEL_DIR = Path("outputs/panels")


# ---------- MODEL CACHE ----------
@st.cache_resource
def load_models():
    from src.vlm.stage1 import Stage1VLM
    from src.vlm.stage2 import Stage2VLM
    return Stage1VLM(), Stage2VLM()

stage1_model, stage2_model = load_models()


# ---------- UI SETUP ----------
st.set_page_config(layout="wide")
st.title("sCT Interpretability Dashboard")

st.divider()
st.subheader("Run / Load Case")

mode = st.radio("Select Mode", ["Existing Case", "New Case"])


# ---------- CASE LIST ----------
cases = [p.name for p in DATA_DIR.iterdir() if p.is_dir()]

if not cases:
    st.warning("No precomputed cases found. You can still run a new case.")


# ---------- MODE HANDLING ----------
case_id = None

if mode == "Existing Case":
    if cases:
        case_id = st.selectbox("Select Case", cases)
    else:
        st.info("No existing cases available.")

else:
    case_id = st.text_input("Enter Case ID")

    ct_path = st.text_input("Path to CT (.mha)")
    mri_path = st.text_input("Path to MRI (.mha)")
    sct_path = st.text_input("Path to sCT (.mha)")

    run_btn = st.button("Run Pipeline ▶️")

    if run_btn:
        if not case_id or not case_id.strip():
            st.error("Case ID required")
            st.stop()

        if not all([ct_path, mri_path, sct_path]):
            st.error("All paths required")
            st.stop()

        case_dir = DATA_DIR / case_id

        if case_dir.exists():
            st.warning("Case already exists. Overwriting files...")

        case_dir.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy(ct_path, case_dir / "ct_preproc.mha")
            shutil.copy(mri_path, case_dir / "mr_preproc.mha")
            shutil.copy(sct_path, case_dir / "sct.mha")
        except Exception as e:
            st.error(f"File copy failed: {e}")
            st.stop()

        # with st.spinner("Running pipeline... (this may take ~30-60s)"):
        #     try:
        #         run_pipeline_for_case(case_id, case_dir, stage1_model, stage2_model)
        #     except Exception as e:
        #         st.error(f"Pipeline failed: {e}")
        #         st.stop()
        import traceback

        with st.spinner("Running pipeline... (this may take ~30-60s)"):
            try:
                run_pipeline_for_case(case_id, case_dir, stage1_model, stage2_model)
            except Exception as e:
                st.error(f"Pipeline failed: {e}")

                # FULL traceback in terminal
                traceback.print_exc()

                # also show in UI (optional but useful)
                st.text(traceback.format_exc())

                st.stop()

        st.success("Pipeline completed!")
        st.rerun()


# ---------- LOAD DATA (ONLY IF CASE EXISTS) ----------
if case_id:

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
        panel_np = np.array(image)
        panel_vis = visualize_panel(panel_np)
        st.image(panel_vis, use_container_width=True)
    else:
        st.info("No image available for this case.")


    # ---------- METRICS ----------
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
