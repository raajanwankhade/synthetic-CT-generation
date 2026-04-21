from src.io.load import load_case
from src.slicing.selection import select_axial_slices
from src.slicing.extract import extract_axial_slices
from src.panels.grid import make_grid
from src.io.save import save_image

case_path = "data/1HNA008"

case = load_case(case_path)

indices = select_axial_slices(case["ct"], num_slices=12)
slices = extract_axial_slices(case, indices)

rows, cols = 3, 4
size = 128

ct_grid = make_grid(slices["ct"], rows, cols, size)
sct_grid = make_grid(slices["sct"], rows, cols, size)
mri_grid = make_grid(slices["mri"], rows, cols, size)
gradcam_grid = make_grid(slices["gradcam"], rows, cols, size)

save_image(ct_grid, "outputs/test/ct.png")
save_image(sct_grid, "outputs/test/sct.png")
save_image(mri_grid, "outputs/test/mri.png")
save_image(gradcam_grid, "outputs/test/gradcam.png")

print("Saved grid images to outputs/test/")
