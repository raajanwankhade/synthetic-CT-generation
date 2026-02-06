import numpy as np

from metrics.image import ImageMetrics


if __name__ == "__main__":
    metrics = ImageMetrics()
    gt = np.zeros((8, 8, 8), dtype=np.float32)
    pred = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.ones((8, 8, 8), dtype=np.float32)
    print(metrics.score_patient(gt, pred, mask))
