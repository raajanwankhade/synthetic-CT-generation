import cv2
from pathlib import Path


def save_image(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(path), img)
