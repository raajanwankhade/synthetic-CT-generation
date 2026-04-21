import re


def extract_metric_block(text, metric_name):
    """
    Extracts the full block for a metric (MAE / PSNR / MS-SSIM)
    """
    pattern = rf"####\s*{metric_name}.*?(?=####|### SUMMARY|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    return match.group(0) if match else ""


def extract_maps_to(block):
    """
    Extract 'Maps to:' from a block
    """
    match = re.search(r"Maps to:\s*(.*)", block, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def parse_stage2_output(text):
    result = {
        "mae": "",
        "psnr": "",
        "ms_ssim": "",
        "summary": ""
    }

    # --- Extract metric blocks ---
    mae_block = extract_metric_block(text, "MAE")
    psnr_block = extract_metric_block(text, "PSNR")
    ssim_block = extract_metric_block(text, "MS-SSIM")

    # --- Extract "Maps to" ---
    result["mae"] = extract_maps_to(mae_block)
    result["psnr"] = extract_maps_to(psnr_block)
    result["ms_ssim"] = extract_maps_to(ssim_block)

    # --- Extract summary ---
    summary_match = re.search(r"### SUMMARY:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if summary_match:
        result["summary"] = summary_match.group(1).strip()

    return result