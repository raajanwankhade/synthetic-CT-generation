from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk

from utils.constants import CT_FILENAME, MASK_FILENAME


def is_patient_dir(path: Path, modality: str) -> bool:
    if not path.is_dir():
        return False
    return (
        (path / CT_FILENAME).exists()
        and (path / f"{modality}.mha").exists()
        and (path / MASK_FILENAME).exists()
    )


def read_image_to_array(path: Union[str, Path]) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
