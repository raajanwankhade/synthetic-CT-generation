import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset
from pathlib import Path
from tqdm.auto import tqdm
from collections import OrderedDict

class MRCTDataset(Dataset):
    def __init__(
        self,
        root,
        modality="mr",
        patch_size=(256, 256),
        num_adjacent_slices=0,
        transform=None,
        max_retries=50,
        foreground_threshold=0.5,
        cache_size=0,
    ):
        """
        Args:
            root: path to folder like 'train/Task1/AB'
            modality: 'mr' or 'cbct' (input modality)
            patch_size: tuple (H, W)
            num_adjacent_slices: e.g., 3 → (i-1, i, i+1)
            transform: optional callable
        """
        self.root = Path(root)
        self.patch_size = patch_size
        if num_adjacent_slices < 0:
            raise ValueError("num_adjacent_slices must be >= 0")

        # Convention:
        # - 0 => pure 2D (single slice, 1 channel)
        # - odd numbers (1,3,5,...) => 2.5D context (stacked slices)
        if num_adjacent_slices == 0:
            self.num_adjacent = 1
        else:
            if num_adjacent_slices % 2 == 0:
                raise ValueError("num_adjacent_slices must be 0 or odd (e.g., 0, 1, 3, 5)")
            self.num_adjacent = num_adjacent_slices

        self.num_adjacent_slices = int(num_adjacent_slices)
        self.transform = transform
        self.modality = modality.lower()
        self.max_retries = max_retries
        self.foreground_threshold = float(foreground_threshold)
        self.cache_size = int(cache_size)
        self._patient_cache = OrderedDict()
        assert self.modality in ["mr", "cbct"], "modality must be 'mr' or 'cbct'"

        # Collect valid patient folders (ignore /overviews/)
        self.patients = []
        for p in sorted(self.root.glob("*")):
            if p.is_dir() and p.name.lower() != "overviews":
                ct_path = p / "ct.mha"
                mod_path = p / f"{self.modality}.mha"
                mask_path = p / "mask.mha"
                if ct_path.exists() and mod_path.exists() and mask_path.exists():
                    self.patients.append(p)

        # --- MODIFIED SECTION START ---
        # Collect all (patient, slice_idx) pairs *that have a non-empty mask*
        self.samples = []
        for p in tqdm(self.patients, desc="Loading Data"):
            # We load the mask to check for non-empty slices
            mask = sitk.GetArrayFromImage(sitk.ReadImage(str(p / "mask.mha")))
            
            for i in range(mask.shape[0]):  # iterate over axial slices
                # Check if the mask slice has any non-zero pixels (i.e., has foreground)
                if np.any(mask[i]):
                    self.samples.append((p, i))
        
        if not self.samples:
            print("Warning: No valid samples found. Check dataset and mask integrity.")
        # --- MODIFIED SECTION END ---

    def _load_patient_arrays(self, patient_path: Path):
        """Load CT, modality, and mask arrays with an optional small LRU cache.

        Note: When using DataLoader with multiple workers, each worker process will
        have its own dataset instance/cache.
        """
        key = str(patient_path)
        if self.cache_size > 0:
            cached = self._patient_cache.get(key)
            if cached is not None:
                self._patient_cache.move_to_end(key)
                return cached

        ct = sitk.GetArrayFromImage(sitk.ReadImage(str(patient_path / "ct.mha")))
        mod = sitk.GetArrayFromImage(sitk.ReadImage(str(patient_path / f"{self.modality}.mha")))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(patient_path / "mask.mha")))

        value = (ct, mod, mask)
        if self.cache_size > 0:
            self._patient_cache[key] = value
            self._patient_cache.move_to_end(key)
            while len(self._patient_cache) > self.cache_size:
                self._patient_cache.popitem(last=False)
        return value

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_path, slice_idx = self.samples[idx]

        ct, mod, mask = self._load_patient_arrays(patient_path)

        # Handle 2.5D context
        half = self.num_adjacent // 2
        idxs = [max(0, min(ct.shape[0] - 1, slice_idx + i)) for i in range(-half, half + 1)]
        mod_stack = np.stack([mod[i] for i in idxs], axis=0)  # (num_adjacent, H, W)

        ct_slice = ct[slice_idx][None, ...]  # (1, H, W)
        mask_slice = mask[slice_idx][None, ...]

        # --- MODIFIED SECTION START ---

        h, w = ct_slice.shape[1:]
        ph, pw = self.patch_size
        if h < ph or w < pw:
            raise ValueError(f"Patch size {self.patch_size} exceeds slice size {(h, w)} for {patient_path}")

        # Find all valid foreground pixels (where mask > 0)
        # We know this is non-empty because of our check in __init__
        y_indices, x_indices = np.nonzero(mask_slice[0] > 0)

        # Always initialize to a valid random patch (acts as a fallback)
        if h == ph and w == pw:
            y0, x0 = 0, 0
        else:
            y0 = int(np.random.randint(0, h - ph + 1))
            x0 = int(np.random.randint(0, w - pw + 1))

        best_y0, best_x0 = y0, x0
        best_ratio = -1.0
        for _ in range(int(self.max_retries)):
            # --- Strategy: pick a center from the mask (if available) ---
            if len(y_indices) > 0:
                rand_idx = int(np.random.randint(0, len(y_indices)))
                center_y = int(y_indices[rand_idx])
                center_x = int(x_indices[rand_idx])

                y0 = int(np.clip(center_y - ph // 2, 0, h - ph))
                x0 = int(np.clip(center_x - pw // 2, 0, w - pw))

            mask_crop = mask_slice[:, y0:y0 + ph, x0:x0 + pw]
            foreground_ratio = float((mask_crop > 0).mean())
            if foreground_ratio > best_ratio:
                best_ratio = foreground_ratio
                best_y0, best_x0 = y0, x0
            if foreground_ratio >= self.foreground_threshold:
                best_y0, best_x0 = y0, x0
                break

        y0, x0 = best_y0, best_x0
        
        # If loop finishes, we use the last patch
        # Now we create the other crops using the final y0, x0
        mod_crop = mod_stack[:, y0:y0 + ph, x0:x0 + pw]
        ct_crop = ct_slice[:, y0:y0 + ph, x0:x0 + pw]
        
        # Need to re-assign mask_crop if the loop exited early
        # (or just crop it once down here)
        # Let's ensure mask_crop is defined based on the final y0, x0
        mask_crop = mask_slice[:, y0:y0 + ph, x0:x0 + pw]

        # --- MODIFIED SECTION END ---

        sample = {
            "input": torch.from_numpy(np.ascontiguousarray(mod_crop)).float(),
            "label": torch.from_numpy(np.ascontiguousarray(ct_crop)).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask_crop)).float(),
            "patient": patient_path.name,
            "slice_idx": slice_idx
        }

        if self.transform:
            sample = self.transform(sample)
        return sample
