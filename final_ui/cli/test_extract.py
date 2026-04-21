from src.io.load import load_case
from src.slicing.selection import select_axial_slices
from src.slicing.extract import extract_axial_slices

case_path = "data/1HNA008"

case = load_case(case_path)

indices = select_axial_slices(case["ct"], num_slices=12)
slices = extract_axial_slices(case, indices)

print("Number of slices extracted:", len(slices["ct"]))

print("\nSample slice shapes:")
for k in slices:
    print(k, slices[k][0].shape)
