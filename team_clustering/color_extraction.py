from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .config import TeamClusteringConfig

logger = logging.getLogger(__name__)


def _resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _make_jersey_mask_bgr(crop_bgr: np.ndarray, cfg: TeamClusteringConfig) -> np.ndarray:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    mask = (s >= cfg.hsv_sat_min) & (v >= cfg.hsv_val_min)

    lo = cfg.hsv_exclude_green_hue_lo
    hi = cfg.hsv_exclude_green_hue_hi
    if lo is not None and hi is not None:
        # Exclude hues in [lo, hi] (grass-like greens)
        mask &= ~((h >= lo) & (h <= hi))

    return (mask.astype(np.uint8) * 255)


def _mean_color_in_space(crop_bgr: np.ndarray, mask: np.ndarray | None, space: str) -> np.ndarray:
    if space == "LAB":
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    elif space == "HSV":
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError(f"Unsupported color space: {space}")

    if mask is None:
        return img.reshape(-1, 3).astype(np.float32).mean(axis=0)

    sel = img[mask > 0]
    if sel.size == 0:
        return img.reshape(-1, 3).astype(np.float32).mean(axis=0)
    return sel.astype(np.float32).mean(axis=0)


def _per_crop_kmeans_color(space_img: np.ndarray, mask: np.ndarray, k: int, max_iter: int) -> np.ndarray:
    """
    Mini k-means on pixels inside mask, returns largest cluster center.
    Uses OpenCV kmeans to avoid sklearn dependency in this stage.
    """
    pixels = space_img[mask > 0].reshape(-1, 3).astype(np.float32)
    if pixels.shape[0] < max(50, k * 20):
        return pixels.mean(axis=0) if pixels.shape[0] else np.zeros((3,), dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(max_iter), 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, flags)
    labels = labels.reshape(-1)
    counts = np.bincount(labels, minlength=k)
    dominant = int(np.argmax(counts))
    return centers[dominant].astype(np.float32)


def extract_color(
    crop_bgr: np.ndarray,
    cfg: TeamClusteringConfig | None = None,
    *,
    debug_prefix: str | None = None,
) -> np.ndarray:
    """
    Extract a stable representative jersey color vector from a BGR crop.

    Returns float32 vector of shape (3,).
    """
    if cfg is None:
        cfg = TeamClusteringConfig()

    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros((3,), dtype=np.float32)

    crop_small = _resize_max_side(crop_bgr, cfg.crop_resize_max_side)
    mask = _make_jersey_mask_bgr(crop_small, cfg)

    masked_frac = float(np.count_nonzero(mask)) / float(mask.size) if mask.size else 0.0
    mask_use = None if masked_frac < cfg.min_masked_fraction else mask

    if cfg.per_crop_kmeans_k is not None and cfg.per_crop_kmeans_k >= 2 and mask_use is not None:
        space_img = (
            cv2.cvtColor(crop_small, cv2.COLOR_BGR2LAB)
            if cfg.color_space == "LAB"
            else cv2.cvtColor(crop_small, cv2.COLOR_BGR2HSV)
        )
        vec = _per_crop_kmeans_color(space_img, mask_use, int(cfg.per_crop_kmeans_k), int(cfg.per_crop_kmeans_max_iter))
    else:
        vec = _mean_color_in_space(crop_small, mask_use, cfg.color_space)

    if cfg.debug and debug_prefix and cfg.debug_save_masks:
        out_dir = cfg.debug_path()
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(out_dir) / f"{debug_prefix}_mask.png"), mask)
        if cfg.debug_save_crops:
            cv2.imwrite(str(Path(out_dir) / f"{debug_prefix}_crop.png"), crop_small)

    return vec.astype(np.float32)

