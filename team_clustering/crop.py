from __future__ import annotations

import logging
from typing import Sequence, Tuple

import numpy as np

from .config import TeamClusteringConfig

logger = logging.getLogger(__name__)


BBoxXYWH = Tuple[int, int, int, int]


def _clip_bbox_xywh(bbox: Sequence[float | int], w_img: int, h_img: int) -> BBoxXYWH | None:
    if bbox is None or len(bbox) != 4:
        return None
    x, y, w, h = (int(round(float(b))) for b in bbox)
    if w <= 1 or h <= 1:
        return None

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)
    if x1 - x0 <= 1 or y1 - y0 <= 1:
        return None
    return x0, y0, x1 - x0, y1 - y0


def crop_player(image: np.ndarray, bbox: Sequence[float | int], cfg: TeamClusteringConfig | None = None) -> np.ndarray | None:
    """
    Crop a player's torso region from an image using an input bbox (x, y, w, h).
    The crop is narrowed horizontally and focused vertically to reduce grass/background.

    Returns None if the bbox is invalid or the crop is empty.
    """
    if cfg is None:
        cfg = TeamClusteringConfig()

    if image is None or image.size == 0:
        return None

    h_img, w_img = image.shape[:2]
    clipped = _clip_bbox_xywh(bbox, w_img=w_img, h_img=h_img)
    if clipped is None:
        logger.debug("Invalid bbox for cropping: %s", bbox)
        return None

    x, y, w, h = clipped

    # Torso vertical range.
    y0 = y + int(round(h * float(cfg.torso_y0)))
    y1 = y + int(round(h * float(cfg.torso_y1)))
    y0 = max(y, min(y0, y + h - 1))
    y1 = max(y0 + 1, min(y1, y + h))

    # Narrow horizontally to avoid background at sides.
    x_pad = int(round(w * float(cfg.torso_x_pad)))
    x0 = x + x_pad
    x1 = x + w - x_pad
    x0 = max(x, min(x0, x + w - 1))
    x1 = max(x0 + 1, min(x1, x + w))

    crop = image[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return None
    return crop

