import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


class TSPseudoLabelCache:
    """
    Stores per-patient TS segmentation volumes (uint8, full 3D, CT coordinate space).
    Updated once per TS cycle after TotalSegmentator runs on the full stitched sCT.
    Queried patch-by-patch during regular training to supply pseudo-labels for
    the surrogate segmentation head.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_binary: bool = True,
        selected_classes: Optional[List[int]] = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.use_binary = use_binary
        self.selected_classes = selected_classes
        self._volumes: Dict[str, np.ndarray] = {}

    @property
    def is_populated(self) -> bool:
        return len(self._volumes) > 0

    @property
    def n_classes(self) -> int:
        if self.use_binary:
            return 1
        if self.selected_classes is not None:
            return len(self.selected_classes)
        return 104

    def update(self, patient_name: str, seg_volume: np.ndarray) -> None:
        """
        Store a TS segmentation volume for one patient.

        seg_volume must already be in training CT coordinate space
        (i.e., resampled from MRI space if MRI and CT grids differ).
        Shape: (D, H, W), dtype uint8.
        """
        if self.use_binary:
            vol = (seg_volume > 0).astype(np.uint8)
        elif self.selected_classes is not None:
            vol = np.zeros_like(seg_volume, dtype=np.uint8)
            for new_idx, cls_idx in enumerate(self.selected_classes):
                vol[seg_volume == cls_idx] = new_idx + 1
        else:
            vol = seg_volume.astype(np.uint8)

        self._volumes[patient_name] = vol

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_dir / f"{patient_name}.npy", vol)

    def get_patch(
        self,
        patient_name: str,
        patch_start: Tuple[int, int, int],
        patch_shape: Tuple[int, int, int],
    ) -> Optional[torch.Tensor]:
        """
        Return a float32 tensor of shape (n_classes, D, H, W) for the given
        patch location, or None if the patient is not cached or the patch is
        out of bounds (e.g. a boundary patch near a volume edge).
        """
        vol = self._volumes.get(patient_name)
        if vol is None:
            return None

        z0, y0, x0 = patch_start
        d, h, w = patch_shape
        vd, vh, vw = vol.shape

        if z0 < 0 or y0 < 0 or x0 < 0 or (z0 + d) > vd or (y0 + h) > vh or (x0 + w) > vw:
            return None

        patch_label = vol[z0:z0 + d, y0:y0 + h, x0:x0 + w]
        label_t = torch.from_numpy(patch_label).long()

        if self.use_binary:
            return (label_t > 0).float().unsqueeze(0)

        one_hot = torch.zeros((self.n_classes, d, h, w), dtype=torch.float32)
        for c in range(self.n_classes):
            one_hot[c] = (label_t == (c + 1)).float()
        return one_hot