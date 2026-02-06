from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray]


def extract_patches_sequentially(
    image: ArrayLike,
    patch_size: Tuple[int, int],
    stride: Optional[Tuple[int, int]] = None,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]], Tuple[int, int]]:
    """
    Extract patches in a deterministic raster-scan order.

    Args:
        image: numpy array with shape (H, W) or (C, H, W)
        patch_size: (ph, pw)
        stride: (sh, sw). If None, defaults to patch_size (non-overlapping).

    Returns:
        patches: list of patches (same shape as input patch)
        coords: list of (y0, x0) top-left coordinates for each patch
        image_hw: (H, W) original spatial size
    """
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D (H,W) or 3D (C,H,W)")

    if image.ndim == 2:
        h, w = image.shape
        c_first = False
    else:
        _, h, w = image.shape
        c_first = True

    ph, pw = patch_size
    if stride is None:
        sh, sw = ph, pw
    else:
        sh, sw = stride

    if ph <= 0 or pw <= 0 or sh <= 0 or sw <= 0:
        raise ValueError("patch_size and stride must be positive")
    if ph > h or pw > w:
        raise ValueError(f"patch_size {patch_size} exceeds image size {(h, w)}")

    ys = list(range(0, max(h - ph + 1, 1), sh))
    xs = list(range(0, max(w - pw + 1, 1), sw))
    if ys[-1] != h - ph:
        ys.append(h - ph)
    if xs[-1] != w - pw:
        xs.append(w - pw)

    patches: List[np.ndarray] = []
    coords: List[Tuple[int, int]] = []

    for y0 in ys:
        for x0 in xs:
            if c_first:
                patch = image[:, y0 : y0 + ph, x0 : x0 + pw]
            else:
                patch = image[y0 : y0 + ph, x0 : x0 + pw]
            patches.append(patch)
            coords.append((y0, x0))

    return patches, coords, (h, w)


def stitch_patches_array(
    patches: Sequence[np.ndarray],
    coords: Sequence[Tuple[int, int]],
    output_shape: Tuple[int, int],
) -> np.ndarray:
    if len(patches) != len(coords):
        raise ValueError("patches and coords must have the same length")
    if not patches:
        raise ValueError("patches is empty")

    h, w = output_shape
    out = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    for patch, (y0, x0) in zip(patches, coords):
        if patch.ndim == 3 and patch.shape[0] == 1:
            patch = patch[0]
        if patch.ndim != 2:
            raise ValueError("patch must be 2D or (1,H,W)")
        ph, pw = patch.shape
        out[y0 : y0 + ph, x0 : x0 + pw] += patch
        weight[y0 : y0 + ph, x0 : x0 + pw] += 1.0

    weight = np.where(weight == 0, 1.0, weight)
    return out / weight


def make_adjacent_stack(volume: np.ndarray, slice_idx: int, num_adjacent_slices: int) -> np.ndarray:
    """
    Build a 2.5D stack similar to dataset_class.MRCTDataset.

    Args:
        volume: (D, H, W)
        slice_idx: target slice index
        num_adjacent_slices: 0 for 2D, or odd numbers (1,3,5,...)

    Returns:
        stack with shape (C, H, W)
    """
    if num_adjacent_slices < 0:
        raise ValueError("num_adjacent_slices must be >= 0")
    if num_adjacent_slices == 0:
        return volume[slice_idx][None, ...]
    if num_adjacent_slices % 2 == 0:
        raise ValueError("num_adjacent_slices must be 0 or odd (e.g., 0, 1, 3, 5)")

    half = num_adjacent_slices // 2
    idxs = [max(0, min(volume.shape[0] - 1, slice_idx + i)) for i in range(-half, half + 1)]
    return np.stack([volume[i] for i in idxs], axis=0)


def extract_patches_sequentially_3d(
    image: ArrayLike,
    patch_size: Tuple[int, int, int],
    stride: Optional[Tuple[int, int, int]] = None,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]], Tuple[int, int, int]]:
    """
    Extract 3D patches in a deterministic raster-scan order.

    Args:
        image: numpy array with shape (D, H, W) or (C, D, H, W)
        patch_size: (pd, ph, pw)
        stride: (sd, sh, sw). If None, defaults to patch_size (non-overlapping).

    Returns:
        patches: list of patches
        coords: list of (z0, y0, x0) top-left coordinates for each patch
        image_dhw: (D, H, W) original spatial size
    """
    if image.ndim not in (3, 4):
        raise ValueError("image must be 3D (D,H,W) or 4D (C,D,H,W)")

    if image.ndim == 3:
        d, h, w = image.shape
        c_first = False
    else:
        _, d, h, w = image.shape
        c_first = True

    pd, ph, pw = patch_size
    if stride is None:
        sd, sh, sw = pd, ph, pw
    else:
        sd, sh, sw = stride

    if pd <= 0 or ph <= 0 or pw <= 0 or sd <= 0 or sh <= 0 or sw <= 0:
        raise ValueError("patch_size and stride must be positive")
    if pd > d or ph > h or pw > w:
        raise ValueError(f"patch_size {patch_size} exceeds image size {(d, h, w)}")

    zs = list(range(0, max(d - pd + 1, 1), sd))
    ys = list(range(0, max(h - ph + 1, 1), sh))
    xs = list(range(0, max(w - pw + 1, 1), sw))
    if zs[-1] != d - pd:
        zs.append(d - pd)
    if ys[-1] != h - ph:
        ys.append(h - ph)
    if xs[-1] != w - pw:
        xs.append(w - pw)

    patches: List[np.ndarray] = []
    coords: List[Tuple[int, int, int]] = []

    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                if c_first:
                    patch = image[:, z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw]
                else:
                    patch = image[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw]
                patches.append(patch)
                coords.append((z0, y0, x0))

    return patches, coords, (d, h, w)


def stitch_patches_array_3d(
    patches: Sequence[np.ndarray],
    coords: Sequence[Tuple[int, int, int]],
    output_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Stitch 3D patches back into a volume with averaging for overlapping regions.

    Args:
        patches: list of 3D patches (D,H,W) or (1,D,H,W)
        coords: list of (z0, y0, x0) coordinates
        output_shape: (D, H, W) of final volume

    Returns:
        stitched volume (D, H, W)
    """
    if len(patches) != len(coords):
        raise ValueError("patches and coords must have the same length")
    if not patches:
        raise ValueError("patches is empty")

    d, h, w = output_shape
    out = np.zeros((d, h, w), dtype=np.float32)
    weight = np.zeros((d, h, w), dtype=np.float32)

    for patch, (z0, y0, x0) in zip(patches, coords):
        if patch.ndim == 4 and patch.shape[0] == 1:
            patch = patch[0]
        if patch.ndim != 3:
            raise ValueError("patch must be 3D (D,H,W) or (1,D,H,W)")
        pd, ph, pw = patch.shape
        out[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] += patch
        weight[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] += 1.0

    weight = np.where(weight == 0, 1.0, weight)
    return out / weight
