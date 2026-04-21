import numpy as np


def combine_modalities(ct_grid, sct_grid, mri_grid):
    """
    Horizontally concatenate modality grids.

    Returns:
        combined image
    """
    # sanity check
    if not (ct_grid.shape == sct_grid.shape == mri_grid.shape):
        raise ValueError("Grid shapes do not match")

    combined = np.concatenate([ct_grid, sct_grid, mri_grid], axis=1)

    return combined
