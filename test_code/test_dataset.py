import numpy as np

from config import cfg
from data.dataset import MRCTDataset


if __name__ == "__main__":
    root = cfg.paths.root  # <-- change this
    modality = "mr"  # "mr" or "cbct"

    n_samples = 3  # how many samples to print per patch size
    foreground_threshold = 0.5
    cache_size = 1

    print(f"Root: {root}")
    print(f"Modality: {modality} | num_adjacent_slices=3d")

    anatomy = "HN"  # "AB", "HN", or "TH"
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
