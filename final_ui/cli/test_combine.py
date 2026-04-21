from src.io.load import load_case
from src.slicing.selection import select_axial_slices
from src.slicing.extract import extract_axial_slices
from src.panels.grid import make_grid
from src.panels.combine import combine_modalities
from src.io.save import save_image
from src.panels.diff import compute_diff_slices

import numpy as np

case_path = "data/1HNA008"

case = load_case(case_path)

indices = select_axial_slices(case["ct"], num_slices=12)
slices = extract_axial_slices(case, indices)

diff_slices = compute_diff_slices(slices["ct"], slices["sct"])

rows, cols = 3, 4
size = 128

ct_grid = make_grid(slices["ct"], rows, cols, size)
sct_grid = make_grid(slices["sct"], rows, cols, size)
mri_grid = make_grid(slices["mri"], rows, cols, size)
diff_grid = make_grid(diff_slices, rows, cols, size)

# combined = combine_modalities(ct_grid, sct_grid, mri_grid)
combined = np.concatenate(
    [ct_grid, sct_grid, mri_grid, diff_grid],
    axis=1
)

save_image(combined, "outputs/test/combined.png")

print("Saved combined panel at outputs/test/combined.png")
print("Combined shape:", combined.shape)
