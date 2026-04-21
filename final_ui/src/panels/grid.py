import numpy as np
import cv2


def normalize_slice(slice_2d):
    """
    Normalize slice to [0,1] for visualization.
    """
    s = slice_2d.astype(np.float32)
    s_min, s_max = s.min(), s.max()

    if s_max - s_min < 1e-6:
        return np.zeros_like(s)

    return (s - s_min) / (s_max - s_min)


def make_grid(slices, rows, cols, size):
    """
    Create grid image from list of slices.

    Args:
        slices: list of 2D arrays
        rows, cols: grid layout
        size: resize each slice to (size, size)

    Returns:
        uint8 image
    """
    canvas = np.zeros((rows * size, cols * size), dtype=np.float32)

    for idx, slc in enumerate(slices[:rows * cols]):
        r = idx // cols
        c = idx % cols

        slc = normalize_slice(slc)
        slc = cv2.resize(slc, (size, size))

        canvas[r*size:(r+1)*size, c*size:(c+1)*size] = slc

    return (canvas * 255).astype(np.uint8)
