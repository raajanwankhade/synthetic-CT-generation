from src.io.load import load_case

case_path = "data/1HNA008"   # change to your actual case

case = load_case(case_path)

print("Loaded successfully")
print("Shape:", case["shape"])
print("Spacing:", case["spacing"])

for k in ["ct", "sct", "mri", "gradcam"]:
    print(f"{k} dtype: {case[k].dtype}, min: {case[k].min()}, max: {case[k].max()}")