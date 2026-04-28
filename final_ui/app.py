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


# ---------- HELPERS ----------
def parse_stage1_sections(text: str) -> dict:
    """
    Splits stage1 assistant response into named sections based on ### headers.
    Returns a dict keyed by section title (lowercased, stripped).
    Any leading text before the first ### is stored under 'preamble'.
    """
    import re
    sections = {}
    parts = re.split(r"(###\s+[^\n]+)", text)

    # parts alternates: [preamble, header, body, header, body, ...]
    preamble = parts[0].strip()
    if preamble:
        sections["preamble"] = preamble

    for i in range(1, len(parts) - 1, 2):
        header = parts[i].strip().lstrip("#").strip()   # e.g. "STRUCTURES IN CT:"
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections[header.lower().rstrip(":")] = body

    return sections


def extract_assistant_response(text: str) -> str:
    """
    The .txt files store the full conversation history:
        system / user prompt / assistant response
    This extracts only the final assistant turn.
    """
    # Split on the literal role marker and take everything after the last one
    parts = text.split("\nassistant\n")
    if len(parts) > 1:
        return parts[-1].strip()
    # Fallback: return as-is if format is unexpected
    return text.strip()


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

        import traceback

        with st.spinner("Running pipeline... (this may take ~30-60s)"):
            try:
                run_pipeline_for_case(case_id, case_dir, stage1_model, stage2_model)
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                traceback.print_exc()
                st.text(traceback.format_exc())
                st.stop()

        st.success("Pipeline completed!")
        st.rerun()


# ---------- LOAD DATA ----------
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


    # ---------- LOAD TEXT OUTPUTS ----------
    stage1_text = None
    stage2_text = None
    parsed = None

    stage1_path = Path(f"outputs/stage1/{case_id}.txt")
    stage2_path = Path(f"outputs/stage2/{case_id}.txt")

    if stage1_path.exists():
        raw_stage1 = stage1_path.read_text()
        stage1_text = extract_assistant_response(raw_stage1)

    if stage2_path.exists():
        stage2_text = stage2_path.read_text()
        parsed = parse_stage2_output(stage2_text)


    # ---------- IMAGE ----------
    st.subheader("CT | sCT | MRI | DIFF")

    if image is not None:
        panel_np = np.array(image)
        panel_vis = visualize_panel(panel_np)
        st.image(panel_vis, use_container_width=True)
    else:
        st.info("No image available for this case.")


    st.divider()


    # =========================
    # METRICS STRIP (full width)
    # =========================

    st.subheader("Metrics")

    if metrics:
        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("MAE", f"{metrics['mae']:.4f}")
            if parsed and parsed.get("mae"):
                st.caption(parsed["mae"])

        with m2:
            st.metric("PSNR", f"{metrics['psnr']:.4f}")
            if parsed and parsed.get("psnr"):
                st.caption(parsed["psnr"])

        with m3:
            st.metric("MS-SSIM", f"{metrics['ms_ssim']:.4f}")
            if parsed and parsed.get("ms_ssim"):
                st.caption(parsed["ms_ssim"])
    else:
        st.info("Metrics not available")

    st.divider()

    # =========================
    # STAGE 1 ANALYSIS — split sections
    # =========================

    st.subheader("Detailed Image Analysis (Stage 1)")

    if stage1_text:
        s1 = parse_stage1_sections(stage1_text)

        # Any preamble line (e.g. "Based on the provided panels...")
        if s1.get("preamble"):
            st.caption(s1["preamble"])

        # Find the keys that match our expected sections (order matters)
        def get_section(d, keyword):
            """Case-insensitive partial key match."""
            for k, v in d.items():
                if keyword in k:
                    return k, v
            return None, ""

        ct_key,   ct_body   = get_section(s1, "structures in ct")
        diff_key, diff_body = get_section(s1, "sct vs ct")
        mri_key,  mri_body  = get_section(s1, "mri visibility")

        # Row: CT Structures | sCT vs CT Differences
        col_a, col_b = st.columns(2)
        with col_a:
            if ct_key:
                st.markdown(f"### {ct_key.upper()}:")
                st.markdown(ct_body)
        with col_b:
            if diff_key:
                st.markdown(f"### {diff_key.upper()}:")
                st.markdown(diff_body)

        # Full-width: MRI Visibility
        if mri_key:
            st.markdown(f"### {mri_key.upper()}:")
            st.markdown(mri_body)

        # Remaining sections (if model adds extras)
        known = {ct_key, diff_key, mri_key, "preamble"}
        for k, v in s1.items():
            if k not in known and v:
                st.markdown(f"### {k.upper()}:")
                st.markdown(v)

    else:
        st.info("Stage 1 not available")

    st.divider()

    # =========================
    # SUMMARY — full width
    # =========================

    st.subheader("Summary")

    if parsed and parsed.get("summary"):
        st.markdown(parsed["summary"])
    else:
        st.info("Summary not available")