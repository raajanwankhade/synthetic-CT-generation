from src.vlm.stage2 import Stage2VLM
from src.vlm.prompts import stage2_prompt
from src.metrics.loader import load_metrics, get_case_metrics
from pathlib import Path


case_id = "1HNA008"

# Load Stage 1 output
stage1_path = Path(f"outputs/stage1/{case_id}.txt")
stage1_text = stage1_path.read_text()

# Load metrics
df = load_metrics("data/per_patient_metrics.csv")
metrics = get_case_metrics(df, case_id)

# Build prompt
prompt = stage2_prompt(stage1_text, metrics)

# Run model
model = Stage2VLM()
result = model.generate(prompt)

# Save output
output_path = Path(f"outputs/stage2/{case_id}.txt")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    f.write(result)

print("Saved Stage 2 output to:", output_path)
