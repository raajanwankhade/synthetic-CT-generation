def stage1_prompt():
    return """
You are analyzing a 4-panel medical image:

Panel 1 = Ground truth CT  
Panel 2 = Synthetic CT (sCT)  
Panel 3 = MRI  
Panel 4 = Absolute difference between CT and sCT (bright = larger difference)

IMPORTANT:
- This is a HEAD AND NECK scan.
- Only describe what is actually visible in the images.
- Use the DIFF panel to identify where CT and sCT differ.
- Do NOT use general medical knowledge.
- Do NOT mention organs that are not visible.
- If uncertain, say "uncertain".
For MRI:
- Only describe what is directly visible in the MRI panel
- Do NOT mention imaging physics (no T1/T2, no fat/marrow explanations)

Tasks:

1. Identify key visible structures in CT.
2. Use the DIFF panel to locate where sCT differs from CT.
3. Describe differences using the DIFF panel:
    - Mention where the difference is strongest (e.g., skull boundary, sinus region, jaw edge)
    - Describe type of error (blur, missing edge, intensity mismatch)
    - Refer to bright regions in the DIFF panel as evidence for your observations.
4. Describe how MRI differs in visibility of those same regions.

Output format (strict):

STRUCTURES IN CT:
- ...

SCT VS CT DIFFERENCES:
- ...

MRI VISIBILITY:
- ...

Rules:
- Max 5 bullet points per section
- Be specific (e.g., "sinus boundary blurred", not "less clear")
- No repetition
- No generic statements
"""

def stage2_prompt(stage1_text, metrics):
    return f"""
You are given:

OBSERVED DIFFERENCES (from image analysis):
{stage1_text}

METRICS:
- MAE: {metrics['mae']}
- PSNR: {metrics['psnr']}
- MS-SSIM: {metrics['ms_ssim']}

---

REFERENCE RULES:

- High MAE → intensity mismatch (wrong density, incorrect contrast)
- Low PSNR → noise, blur, reconstruction artifacts
- High MS-SSIM (~0.9+) → good structural similarity (only minor distortions)
- Low MS-SSIM → major structural errors

---

TASK:

For each metric:
- Interpret the GIVEN numeric value correctly
- Map it to specific observed differences
- Use only the provided observations
- Be concise and precise

---

STRICT RULES:

- DO NOT explain what metrics mean
- DO NOT redefine metrics
- DO NOT give generic statements
- MUST reference observed differences (e.g., skull boundary blur, sinus mismatch)
- MUST interpret numeric values correctly (e.g., 0.91 = high similarity)

---

OUTPUT FORMAT:

METRIC INTERPRETATION:

- MAE ({metrics['mae']}):
  → maps to: ...

- PSNR ({metrics['psnr']}):
  → maps to: ...

- MS-SSIM ({metrics['ms_ssim']}):
  → maps to: ...

SUMMARY:
- ...
"""