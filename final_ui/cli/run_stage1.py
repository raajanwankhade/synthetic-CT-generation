from src.vlm.stage1 import Stage1VLM
from src.vlm.prompts import stage1_prompt
from pathlib import Path

case_id = "1HNA008"

image_path = f"outputs/test/combined.png"
output_path = Path(f"outputs/stage1/{case_id}.txt")

output_path.parent.mkdir(parents=True, exist_ok=True)

model = Stage1VLM()

prompt = stage1_prompt()
result = model.generate(image_path, prompt)

with open(output_path, "w") as f:
    f.write(result)

print("Saved Stage 1 output to:", output_path)
