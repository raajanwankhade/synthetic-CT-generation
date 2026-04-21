from pathlib import Path
from src.postprocess.parser import parse_stage2_output

case_id = "1HNA008"

stage2_path = Path(f"outputs/stage2/{case_id}.txt")
text = stage2_path.read_text()

parsed = parse_stage2_output(text)

print("\nParsed Output:\n")

for k, v in parsed.items():
    print(f"{k.upper()}:")
    print(v)
    print("-" * 40)
