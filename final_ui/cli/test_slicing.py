from src.io.load import load_case
from src.slicing.selection import select_axial_slices

case_path = "data/1HNA008"

case = load_case(case_path)
ct = case["ct"]

indices = select_axial_slices(ct, num_slices=12)

print("Selected slice indices:")
print(indices)

print("Total slices:", ct.shape[0])
