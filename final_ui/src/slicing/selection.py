import numpy as np


def select_axial_slices(volume, num_slices):
    """
    Select evenly spaced axial slices, avoiding extreme ends.

    Args:
        volume: (Z, H, W)
        num_slices: int

    Returns:
        list of indices
    """
    z = volume.shape[0]

    # avoid extreme 10% on both ends
    start = int(0.1 * z)
    end = int(0.9 * z)

    if end <= start:
        raise ValueError("Volume too small for slice selection")

    indices = np.linspace(start, end, num_slices).astype(int)

    return indices.tolist()
