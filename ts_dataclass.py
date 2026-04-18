from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader, Dataset


def extract_patches_sequentially_3d(
    image: np.ndarray,
    patch_size: Tuple[int, int, int],
    stride: Optional[Tuple[int, int, int]] = None,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]], Tuple[int, int, int]]:
    """
    Extract 3D patches in deterministic raster-scan order (z, y, x).

    Args:
        image: (D,H,W) or (C,D,H,W)
        patch_size: (pd, ph, pw)
        stride: (sd, sh, sw). Defaults to non-overlapping patches.

    Returns:
        patches: list of extracted patches
        coords:  list of (z0, y0, x0) top-left-front corners
        output_shape: (D, H, W) of the full volume
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
    sd, sh, sw = stride if stride is not None else patch_size

    if pd <= 0 or ph <= 0 or pw <= 0 or sd <= 0 or sh <= 0 or sw <= 0:
        raise ValueError("patch_size and stride values must be positive")
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
                    patch = image[:, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
                else:
                    patch = image[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
                patches.append(patch)
                coords.append((z0, y0, x0))

    return patches, coords, (d, h, w)


def stitch_patches_array_3d(
    patches: Sequence[np.ndarray],
    coords: Sequence[Tuple[int, int, int]],
    output_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Stitch 3D patches back into one volume, averaging overlaps."""
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
            raise ValueError("patch must be (D,H,W) or (1,D,H,W)")
        pd, ph, pw = patch.shape
        out[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw] += patch
        weight[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw] += 1.0

    weight = np.where(weight == 0, 1.0, weight)
    return out / weight


def _is_patient_dir(path: Path, modality: str) -> bool:
    if not path.is_dir():
        return False
    has_ct = (path / "ct.mha").exists() or (path / "ct_preproc.mha").exists()
    has_mod = (path / f"{modality}.mha").exists() or (path / f"{modality}_preproc.mha").exists()
    return has_ct and has_mod and (path / "mask.mha").exists()


def _discover_patients(root: Path, modality: str) -> List[Path]:
    """
    Discover patient folders from root.

    Supported layouts:
    - patient folder itself
    - region folder containing patient folders
    - task folder containing region folders which contain patient folders
    """
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    if _is_patient_dir(root, modality):
        return [root]

    child_dirs = [p for p in sorted(root.glob("*")) if p.is_dir() and p.name.lower() != "overviews"]
    if any(_is_patient_dir(p, modality) for p in child_dirs):
        return [p for p in child_dirs if _is_patient_dir(p, modality)]

    patients: List[Path] = []
    for region in child_dirs:
        for p in sorted(region.glob("*")):
            if p.is_dir() and p.name.lower() != "overviews" and _is_patient_dir(p, modality):
                patients.append(p)

    return patients


class TSPatch3DDataset(Dataset):
    """
    Sequential 3D patch dataset for the TS inference phase.

    Selects a fraction of patients and enumerates all 3D patches for each
    in deterministic raster-scan order (z, y, x). Patches are extracted from
    the modality (MRI) volume; coordinates are in MRI voxel space.
    """

    def __init__(
        self,
        dataset_root: str,
        fraction: float = 1.0,
        modality: str = "mr",
        patch_size: Tuple[int, int, int] = (32, 192, 160),
        stride: Optional[Tuple[int, int, int]] = None,
        use_preproc_files: bool = False,
        seed: int = 42,
        cache_size: int = 2,
        include_patient_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(dataset_root)
        self.modality = modality.lower()
        self.patch_size = tuple(int(x) for x in patch_size)
        self.stride = tuple(int(x) for x in stride) if stride is not None else None
        self.use_preproc_files = bool(use_preproc_files)
        self.seed = int(seed)
        self.cache_size = int(cache_size)
        self.include_patient_names = set(include_patient_names) if include_patient_names is not None else None

        if len(self.patch_size) != 3:
            raise ValueError("patch_size must be a 3-tuple (D, H, W)")
        if self.stride is not None and len(self.stride) != 3:
            raise ValueError("stride must be None or a 3-tuple (sd, sh, sw)")
        if not (0.0 < float(fraction) <= 1.0):
            raise ValueError("fraction must be in (0, 1]")

        all_patients = _discover_patients(self.root, self.modality)
        if self.include_patient_names is not None:
            all_patients = [p for p in all_patients if p.name in self.include_patient_names]
        if not all_patients:
            raise ValueError("No valid patient folders found under dataset_root.")

        rng = np.random.default_rng(self.seed)
        perm = np.arange(len(all_patients))
        rng.shuffle(perm)
        n_select = max(1, int(np.ceil(len(all_patients) * float(fraction))))
        chosen_idx = np.sort(perm[:n_select])
        self.patients = [all_patients[i] for i in chosen_idx]

        self.patient_records: List[Dict[str, object]] = []
        self.patient_name_to_idx: Dict[str, int] = {}
        self.index_map: List[Tuple[int, int]] = []
        self.patient_to_dataset_indices: Dict[str, List[int]] = {}

        for patient_idx, patient_path in enumerate(self.patients):
            mod_arr = self._read_modality_array(patient_path)
            if mod_arr.ndim != 3:
                raise ValueError(
                    f"Modality volume must be (D,H,W), got shape={mod_arr.shape} for {patient_path}"
                )

            _, coords, out_shape = extract_patches_sequentially_3d(
                mod_arr, patch_size=self.patch_size, stride=self.stride,
            )

            record = {
                "patient_path": patient_path,
                "coords": coords,
                "shape": tuple(int(x) for x in out_shape),
            }
            self.patient_records.append(record)
            self.patient_name_to_idx[patient_path.name] = patient_idx

            dataset_indices: List[int] = []
            for patch_idx in range(len(coords)):
                self.index_map.append((patient_idx, patch_idx))
                dataset_indices.append(len(self.index_map) - 1)
            self.patient_to_dataset_indices[patient_path.name] = dataset_indices

        self._array_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._cache_order: List[str] = []
        self._seg_cache: Dict[str, np.ndarray] = {}
        self._seg_cache_order: List[str] = []
        self._ref_img_cache: Dict[str, sitk.Image] = {}
        self._ref_img_cache_order: List[str] = []

    def _read_modality_array(self, patient_path: Path) -> np.ndarray:
        mod_name = f"{self.modality}_preproc.mha" if self.use_preproc_files else f"{self.modality}.mha"
        return sitk.GetArrayFromImage(sitk.ReadImage(str(patient_path / mod_name)))

    def _cache_put(self, cache: Dict[str, Any], order: List[str], key: str, value: Any) -> None:
        cache[key] = value
        order.append(key)
        if self.cache_size > 0 and len(order) > self.cache_size:
            evict = order.pop(0)
            cache.pop(evict, None)

    def get_patient_path(self, patient_name: str) -> Path:
        patient_idx = self.patient_name_to_idx.get(patient_name)
        if patient_idx is None:
            raise ValueError(f"Patient {patient_name} is not part of this dataset")
        return cast(Path, self.patient_records[patient_idx]["patient_path"])

    def get_patient_reference_image(
        self,
        patient_name: str,
        use_preproc_override: Optional[bool] = None,
    ) -> sitk.Image:
        """Return the CT reference image (used as the training coordinate space target)."""
        patient_path = self.get_patient_path(patient_name)
        use_preproc = self.use_preproc_files if use_preproc_override is None else bool(use_preproc_override)
        key = f"{patient_path}|ct|preproc={int(use_preproc)}"
        cached = self._ref_img_cache.get(key)
        if cached is not None:
            return cached
        ref_name = "ct_preproc.mha" if use_preproc else "ct.mha"
        ref_img = sitk.ReadImage(str(patient_path / ref_name))
        if self.cache_size > 0:
            self._cache_put(self._ref_img_cache, self._ref_img_cache_order, key, ref_img)
        return ref_img

    def get_patient_modality_reference_image(self, patient_name: str) -> sitk.Image:
        """
        Return the modality (MRI) reference image for the given patient.

        The stitched sCT is assembled in MRI voxel space (since the TS dataset
        extracts patches from the MRI). This reference image carries the correct
        spatial metadata (origin, spacing, direction) to tag that stitched volume
        before passing it to TotalSegmentator.
        """
        patient_path = self.get_patient_path(patient_name)
        mod_name = f"{self.modality}_preproc.mha" if self.use_preproc_files else f"{self.modality}.mha"
        key = f"{patient_path}|mod|preproc={int(self.use_preproc_files)}"
        cached = self._ref_img_cache.get(key)
        if cached is not None:
            return cached
        ref_img = sitk.ReadImage(str(patient_path / mod_name))
        if self.cache_size > 0:
            self._cache_put(self._ref_img_cache, self._ref_img_cache_order, key, ref_img)
        return ref_img

    def get_patient_segmentation(self, patient_name: str) -> Optional[np.ndarray]:
        """Return the ground-truth segmentation volume (CT space) or None if absent."""
        patient_path = self.get_patient_path(patient_name)
        key = str(patient_path)
        cached = self._seg_cache.get(key)
        if cached is not None:
            return cached
        seg_path = patient_path / "segmentation.nii.gz"
        if not seg_path.exists():
            return None
        seg = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path)))
        if self.cache_size > 0:
            self._cache_put(self._seg_cache, self._seg_cache_order, key, seg)
        return seg

    def _load_patient_arrays(self, patient_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        key = str(patient_path)
        cached = self._array_cache.get(key)
        if cached is not None:
            return cached
        ct_name = "ct_preproc.mha" if self.use_preproc_files else "ct.mha"
        mod_name = f"{self.modality}_preproc.mha" if self.use_preproc_files else f"{self.modality}.mha"
        ct = sitk.GetArrayFromImage(sitk.ReadImage(str(patient_path / ct_name)))
        mod = sitk.GetArrayFromImage(sitk.ReadImage(str(patient_path / mod_name)))
        value = (ct, mod)
        if self.cache_size > 0:
            self._cache_put(self._array_cache, self._cache_order, key, value)
        return value

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Dict[str, object]:
        patient_idx, patch_idx = self.index_map[index]
        record = self.patient_records[patient_idx]
        patient_path = cast(Path, record["patient_path"])
        coords = cast(List[Tuple[int, int, int]], record["coords"])
        out_shape = record["shape"]

        z0, y0, x0 = coords[patch_idx]
        pd, ph, pw = self.patch_size

        ct, mod = self._load_patient_arrays(patient_path)
        mod_patch = mod[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw][None, ...]
        ct_patch = ct[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw][None, ...]

        return {
            "input": torch.from_numpy(np.ascontiguousarray(mod_patch)).float(),
            "label": torch.from_numpy(np.ascontiguousarray(ct_patch)).float(),
            "coord": torch.tensor([z0, y0, x0], dtype=torch.long),
            "patch_start": (int(z0), int(y0), int(x0)),
            "output_shape": torch.tensor(out_shape, dtype=torch.long),
            "patient": patient_path.name,
            "patient_idx": patient_idx,
            "patch_idx": patch_idx,
        }

    @staticmethod
    def stitch_volume(
        patches: Sequence[np.ndarray],
        coords: Sequence[Tuple[int, int, int]],
        output_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        return stitch_patches_array_3d(patches, coords, output_shape)

    def stitch_patient_predictions(
        self, patient_name: str, pred_patches: Sequence[np.ndarray]
    ) -> np.ndarray:
        """Stitch predicted patches for a patient into a full volume (MRI voxel space)."""
        match_idx = self.patient_name_to_idx.get(patient_name)
        if match_idx is None:
            raise ValueError(f"Patient {patient_name} is not part of this dataset")
        record = self.patient_records[match_idx]
        coords = cast(List[Tuple[int, int, int]], record["coords"])
        out_shape = cast(Tuple[int, int, int], record["shape"])
        if len(pred_patches) != len(coords):
            raise ValueError(
                f"pred_patches length mismatch for {patient_name}: "
                f"expected {len(coords)}, got {len(pred_patches)}"
            )
        return stitch_patches_array_3d(pred_patches, coords, out_shape)


def build_ts_dataloader(
    dataset_root: str,
    fraction: float = 1.0,
    modality: str = "mr",
    patch_size: Tuple[int, int, int] = (32, 192, 160),
    stride: Optional[Tuple[int, int, int]] = None,
    use_preproc_files: bool = False,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle: bool = False,
    seed: int = 42,
    cache_size: int = 2,
    include_patient_names: Optional[Sequence[str]] = None,
) -> DataLoader:
    dataset = TSPatch3DDataset(
        dataset_root=dataset_root,
        fraction=fraction,
        modality=modality,
        patch_size=patch_size,
        stride=stride,
        use_preproc_files=use_preproc_files,
        seed=seed,
        cache_size=cache_size,
        include_patient_names=include_patient_names,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


if __name__ == "__main__":
    import os

    dataset_root = "/data/home/sk1022/MajorProject_sCTGen/awesome-synthetic-CT-generation/bhuvanesh/datasets/bhuvaneshsingla/synthrad2025-hn/versions/Experiment"
    modality = os.environ.get("TS_MODALITY", "mr")
    use_preproc_files = bool(int(os.environ.get("TS_USE_PREPROC", "0")))
    patch_size = (32, 192, 160)
    stride = patch_size

    all_patients = _discover_patients(Path(dataset_root), modality)
    test_patients = min(2, len(all_patients))
    if test_patients == 0:
        raise ValueError("No valid patients found for testing")

    fraction = min(1.0, float(test_patients) / float(len(all_patients)))
    loader = build_ts_dataloader(
        dataset_root=dataset_root,
        fraction=fraction,
        modality=modality,
        patch_size=patch_size,
        stride=stride,
        use_preproc_files=use_preproc_files,
        batch_size=1,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        seed=42,
    )

    ds: TSPatch3DDataset = loader.dataset  # type: ignore[assignment]
    print(f"Dataset root: {dataset_root}")
    print(f"Selected patients: {len(ds.patients)}")
    print(f"Total sequential patches: {len(ds)}")

    for patient_path in ds.patients[:2]:
        patient_name = patient_path.name
        pred_patches: List[np.ndarray] = []
        for idx in ds.patient_to_dataset_indices[patient_name]:
            sample = ds[idx]
            pred_patches.append(sample["label"][0].numpy())
        stitched = ds.stitch_patient_predictions(patient_name, pred_patches)
        ct_full, _ = ds._load_patient_arrays(patient_path)
        mae = float(np.mean(np.abs(stitched - ct_full.astype(np.float32))))
        print(
            f"Patient={patient_name} | patches={len(pred_patches)} | "
            f"stitched_shape={tuple(stitched.shape)} | recon_mae={mae:.6f}"
        )