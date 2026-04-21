import numpy as np
import cv2


def visualize_panel(panel, rows=3, cols=4):
    # convert to RGB if needed
    if len(panel.shape) == 2:
        panel = np.stack([panel] * 3, axis=-1)

    panel = panel.copy()

    h, w, _ = panel.shape
    section_w = w // 4

    labels = ["CT", "sCT", "MRI", "DIFF"]

    for i in range(4):
        start = i * section_w
        end = (i + 1) * section_w

        sub = panel[:, start:end]

        cell_h = h // rows
        cell_w = (end - start) // cols

        # grid lines
        for r in range(1, rows):
            y = r * cell_h
            cv2.line(panel, (start, y), (end, y), (255, 255, 255), 1)

        for c in range(1, cols):
            x = start + c * cell_w
            cv2.line(panel, (x, 0), (x, h), (255, 255, 255), 1)

        # label
        cv2.putText(
            panel,
            labels[i],
            (start + 10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return panel