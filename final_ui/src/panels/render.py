import numpy as np
import cv2


def add_grid_lines(img, rows, cols, color=(255, 255, 255), thickness=1):
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)

    h, w, _ = img.shape
    cell_h = h // rows
    cell_w = w // cols

    for i in range(1, rows):
        y = i * cell_h
        cv2.line(img, (0, y), (w, y), color, thickness)

    for j in range(1, cols):
        x = j * cell_w
        cv2.line(img, (x, 0), (x, h), color, thickness)

    return img

def add_labels(panel, labels, font_scale=0.6, thickness=2):
    if len(panel.shape) == 2:
        panel = np.stack([panel] * 3, axis=-1)

    h, w, _ = panel.shape
    section_w = w // len(labels)

    for i, label in enumerate(labels):
        x = i * section_w + 10
        y = 25

        cv2.putText(
            panel,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return panel


def render_combined(ct_grid, sct_grid, mri_grid, diff_grid, rows, cols):
    if len(ct_grid.shape) == 2:
        ct_grid = np.stack([ct_grid] * 3, axis=-1)
        sct_grid = np.stack([sct_grid] * 3, axis=-1)
        mri_grid = np.stack([mri_grid] * 3, axis=-1)
        diff_grid = np.stack([diff_grid] * 3, axis=-1)
    # concat panels
    combined = np.concatenate([ct_grid, sct_grid, mri_grid, diff_grid], axis=1)

    # add grid lines per section
    section_w = combined.shape[1] // 4

    for i in range(4):
        start = i * section_w
        end = (i + 1) * section_w

        combined[:, start:end] = add_grid_lines(
            combined[:, start:end].copy(), rows, cols
        )

    # add labels
    labels = ["CT", "sCT", "MRI", "DIFF"]
    combined = add_labels(combined, labels)

    return combined