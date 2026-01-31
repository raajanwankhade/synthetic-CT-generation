import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from tqdm.auto import tqdm
from collections import OrderedDict
import random

# 3D patch sizes by anatomy
PATCH_SIZE_3D_BY_ANATOMY = {
    "AB": (40, 192, 224),
    "HN": (56, 192, 160),
    "TH": (64, 192, 192),
}

class MRCTDataset(Dataset):
    def __init__(
        self,
        root,
        modality="mr",
        patch_size=(256, 256),
        anatomy=None,
        num_adjacent_slices=0,
        transform=None,
        max_retries=50,
        foreground_threshold=0.5,
        cache_size=0,
    ):
        """
        Args:
            root: path to a region folder like 'train/Task1/AB' OR a task folder like 'train/Task1'
                  (when a task folder is provided, all region subfolders are scanned and combined)
            modality: 'mr' or 'cbct' (input modality)
            patch_size: tuple (H, W)
            num_adjacent_slices: e.g., 3 → (i-1, i, i+1)
            transform: optional callable
        """
        self.root = Path(root)
        self.anatomy = anatomy
        self.is_3d = False

        if isinstance(num_adjacent_slices, str):
            if num_adjacent_slices.lower() != "3d":
                raise ValueError("num_adjacent_slices must be >= 0, odd, or '3d'")
            self.is_3d = True
            if patch_size is None:
                patch_size = PATCH_SIZE_3D_BY_ANATOMY.get(str(anatomy).upper())
        else:
            if num_adjacent_slices < 0:
                raise ValueError("num_adjacent_slices must be >= 0")

        if patch_size is None:
            raise ValueError("patch_size must be provided (or set anatomy=AB/HN/TH for 3D)")

        self.patch_size = tuple(patch_size)
        if len(self.patch_size) == 3:
            self.is_3d = True

        # Convention:
        # - 0 => pure 2D (single slice, 1 channel)
        # - odd numbers (1,3,5,...) => 2.5D context (stacked slices)
        if self.is_3d:
            # For 3D patches, we ignore 2.5D context slicing
            self.num_adjacent = 1
        elif num_adjacent_slices == 0:
            self.num_adjacent = 1
        else:
            if num_adjacent_slices % 2 == 0:
                raise ValueError("num_adjacent_slices must be 0 or odd (e.g., 0, 1, 3, 5)")
            self.num_adjacent = num_adjacent_slices

        self.num_adjacent_slices = num_adjacent_slices
        self.transform = transform
        self.modality = modality.lower()
        self.max_retries = max_retries
        self.foreground_threshold = float(foreground_threshold)
        self.cache_size = int(cache_size)
        self._patient_cache = OrderedDict()
        assert self.modality in ["mr", "cbct"], "modality must be 'mr' or 'cbct'"

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        def _is_patient_dir(d: Path) -> bool:
            if not d.is_dir():
                return False
            return (
                (d / "ct.mha").exists()
                and (d / f"{self.modality}.mha").exists()
                and (d / "mask.mha").exists()
            )

        # Determine whether 'root' is:
        # 1) a patient dir  -> contains ct.mha/mr.mha/mask.mha
        # 2) a region dir   -> contains patient subdirs
        # 3) a task dir     -> contains region subdirs, each containing patient subdirs
        self.region_roots = []
        if _is_patient_dir(self.root):
            # Treat as a single patient
            self.region_roots = [self.root.parent]
        else:
            child_dirs = [p for p in sorted(self.root.glob("*")) if p.is_dir() and p.name.lower() != "overviews"]
            if any(_is_patient_dir(p) for p in child_dirs):
                # Root is a region folder
                self.region_roots = [self.root]
            else:
                # Root is likely a task folder (scan all regions)
                for region in child_dirs:
                    region_children = [p for p in sorted(region.glob("*")) if p.is_dir() and p.name.lower() != "overviews"]
                    if any(_is_patient_dir(p) for p in region_children):
                        self.region_roots.append(region)

        if not self.region_roots:
            raise ValueError(
                "Could not find any valid patient folders under root. "
                "Expected either a region folder with patient subfolders, or a task folder with region subfolders. "
                f"Got root={self.root}"
            )

        # Collect valid patient folders across one or more region roots (ignore /overviews/)
        self.patients = []
        self.patient_region = {}
        if _is_patient_dir(self.root):
            # root is itself a patient directory
            self.patients = [self.root]
            self.patient_region[str(self.root)] = self.root.parent.name
        else:
            for region_root in self.region_roots:
                for p in sorted(region_root.glob("*")):
                    if p.is_dir() and p.name.lower() != "overviews" and _is_patient_dir(p):
                        self.patients.append(p)
                        self.patient_region[str(p)] = region_root.name

        self.regions = sorted(set(self.patient_region.values()))

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

        if self.is_3d:
            # 3D patching
            d, h, w = ct.shape
            pd, ph, pw = self.patch_size
            if d < pd or h < ph or w < pw:
                raise ValueError(
                    f"3D patch size {self.patch_size} exceeds volume size {(d, h, w)} for {patient_path}"
                )

            # Use the current slice_idx as center depth, and pick center (y,x) from mask on that slice
            mask_slice = mask[slice_idx]
            y_indices, x_indices = np.nonzero(mask_slice > 0)

            # Fallback random patch
            z0 = int(np.random.randint(0, d - pd + 1)) if d > pd else 0
            y0 = int(np.random.randint(0, h - ph + 1)) if h > ph else 0
            x0 = int(np.random.randint(0, w - pw + 1)) if w > pw else 0

            best_z0, best_y0, best_x0 = z0, y0, x0
            best_ratio = -1.0
            for _ in range(int(self.max_retries)):
                if len(y_indices) > 0:
                    rand_idx = int(np.random.randint(0, len(y_indices)))
                    center_y = int(y_indices[rand_idx])
                    center_x = int(x_indices[rand_idx])
                    center_z = int(slice_idx)

                    z0 = int(np.clip(center_z - pd // 2, 0, d - pd))
                    y0 = int(np.clip(center_y - ph // 2, 0, h - ph))
                    x0 = int(np.clip(center_x - pw // 2, 0, w - pw))

                mask_crop = mask[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
                foreground_ratio = float((mask_crop > 0).mean())
                if foreground_ratio > best_ratio:
                    best_ratio = foreground_ratio
                    best_z0, best_y0, best_x0 = z0, y0, x0
                if foreground_ratio >= self.foreground_threshold:
                    best_z0, best_y0, best_x0 = z0, y0, x0
                    break

            z0, y0, x0 = best_z0, best_y0, best_x0

            mod_crop = mod[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw][None, ...]
            ct_crop = ct[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw][None, ...]
            mask_crop = mask[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw][None, ...]
        else:
            # Handle 2.5D context
            half = self.num_adjacent // 2
            idxs = [max(0, min(ct.shape[0] - 1, slice_idx + i)) for i in range(-half, half + 1)]
            mod_stack = np.stack([mod[i] for i in idxs], axis=0)  # (num_adjacent, H, W)

            ct_slice = ct[slice_idx][None, ...]  # (1, H, W)
            mask_slice = mask[slice_idx][None, ...]

            h, w = ct_slice.shape[1:]
            ph, pw = self.patch_size
            if h < ph or w < pw:
                raise ValueError(
                    f"Patch size {self.patch_size} exceeds slice size {(h, w)} for {patient_path}"
                )

            # Find all valid foreground pixels (where mask > 0)
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

            mod_crop = mod_stack[:, y0:y0 + ph, x0:x0 + pw]
            ct_crop = ct_slice[:, y0:y0 + ph, x0:x0 + pw]
            mask_crop = mask_slice[:, y0:y0 + ph, x0:x0 + pw]

        sample = {
            "input": torch.from_numpy(np.ascontiguousarray(mod_crop)).float(),
            "label": torch.from_numpy(np.ascontiguousarray(ct_crop)).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask_crop)).float(),
            "patient": patient_path.name,
            "region": self.patient_region.get(str(patient_path), ""),
            "slice_idx": slice_idx
        }

        if self.transform:
            sample = self.transform(sample)
        return sample


def _seed_worker(worker_id: int, seed: int):
    worker_seed = (seed + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(dataset: Dataset, cfg: dict, seed: int, device: torch.device):
    val_split = cfg["val_split"]

    patient_ids = np.array([str(p) for (p, _) in dataset.samples])
    uniq_patients = np.unique(patient_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq_patients)

    n_val_patients = int(len(uniq_patients) * val_split)
    val_patients = set(uniq_patients[:n_val_patients])
    train_patients = set(uniq_patients[n_val_patients:])

    train_idx = np.where(np.isin(patient_ids, list(train_patients)))[0].tolist()
    val_idx = np.where(np.isin(patient_ids, list(val_patients)))[0].tolist()

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    g = torch.Generator()
    g.manual_seed(seed)

    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers_train"],
        pin_memory=pin,
        persistent_workers=(cfg["num_workers_train"] > 0),
        worker_init_fn=lambda wid: _seed_worker(wid, seed),
        generator=g,
        prefetch_factor=cfg["prefetch_factor"],
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers_val"],
        pin_memory=pin,
        persistent_workers=(cfg["num_workers_val"] > 0),
        worker_init_fn=lambda wid: _seed_worker(wid, seed),
        generator=g,
        prefetch_factor=cfg["prefetch_factor"],
    )

    return train_loader, val_loader, train_ds, val_ds, train_patients, val_patients
if __name__ == "__main__":
    # Edit these values manually and run:
    #   python dataset_class.py

    # You can point this either to:
    # - a single region folder: ...\\train\\Task1\\AB
    # - the task folder:        ...\\train\\Task1   (will scan all regions under it)
    from config import cfg
    root = cfg["root"]  # <-- change this
    modality = "mr"  # "mr" or "cbct"
    # num_adjacent_slices = 0  # 0 for 2D, or odd values (1,3,5,...) for 2.5D
    # patch_sizes = [
    #     (256, 256),
    #     (192, 192),
    #     (128, 128),
    #     (64, 64),
    #     (32, 32),
    # ]

    n_samples = 3  # how many samples to print per patch size
    foreground_threshold = 0.5
    cache_size = 1

    print(f"Root: {root}")
    print(f"Modality: {modality} | num_adjacent_slices=3d")

    # 3D test using anatomy-based patch sizes
    anatomy = "AB"  # "AB", "HN", or "TH"
    try:
        ds = MRCTDataset(
            root=root,
            modality=modality,
            patch_size=None,
            anatomy=anatomy,
            num_adjacent_slices="3d",
            max_retries=50,
            foreground_threshold=foreground_threshold,
            cache_size=cache_size,
        )
        print(f"Patients found: {len(ds.patients)}")
        print(f"Regions found: {len(ds.regions)} -> {ds.regions}")
        print(f"Samples found (mask-nonempty slices): {len(ds)}")
        if len(ds) == 0:
            print("No samples to draw from.")
        else:
            rng = np.random.default_rng(0)
            pick = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
            for ds_idx in pick:
                sample = ds[int(ds_idx)]
                x = sample["input"]
                y = sample["label"]
                m = sample["mask"]
                print(
                    f"idx={int(ds_idx)} | patient={sample['patient']} slice={sample['slice_idx']} | "
                    f"region={sample.get('region','')} | "
                    f"input={tuple(x.shape)} label={tuple(y.shape)} mask={tuple(m.shape)}"
                )
        print("OK")
    except Exception as e:
        print(f"FAILED for 3D test: {type(e).__name__}: {e}")
