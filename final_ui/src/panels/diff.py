import numpy as np


def compute_diff_slices(ct_slices, sct_slices):
    """
    Compute absolute difference per slice.
    """
    diff_slices = []

    for ct, sct in zip(ct_slices, sct_slices):
        diff = np.abs(ct.astype(np.float32) - sct.astype(np.float32))
        diff_slices.append(diff)

    return diff_slices
