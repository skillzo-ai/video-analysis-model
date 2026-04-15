"""Serialize ``sv.Detections`` to YOLO txt (normalized xywh) for dataset export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import supervision as sv


def detections_to_yolo_txt(
    detections: sv.Detections,
    img_w: int,
    img_h: int,
) -> str:
    """One line per box: ``class cx cy w h`` (normalized 0–1)."""
    if detections is None or len(detections) == 0:
        return ""

    cls = detections.class_id
    if cls is None:
        return ""

    lines: list[str] = []
    xyxy = detections.xyxy.astype(np.float64)
    iw, ih = float(img_w), float(img_h)
    for i in range(len(detections)):
        x1, y1, x2, y2 = xyxy[i]
        w = max(1e-6, x2 - x1)
        h = max(1e-6, y2 - y1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        lines.append(
            f"{int(cls[i])} {cx / iw:.6f} {cy / ih:.6f} {w / iw:.6f} {h / ih:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def write_classes_txt(names: dict[int, str], path: Path) -> None:
    """``names`` from ``YOLO.model.names`` (indices 0..n-1)."""
    max_id = max(names.keys()) if names else -1
    rows = [names.get(i, str(i)) for i in range(max_id + 1)]
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
