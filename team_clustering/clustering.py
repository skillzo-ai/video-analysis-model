from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from .config import TeamClusteringConfig

logger = logging.getLogger(__name__)


def _kmeans_sklearn(x: np.ndarray, k: int, cfg: TeamClusteringConfig) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans  # type: ignore

    km = KMeans(
        n_clusters=k,
        n_init=cfg.kmeans_n_init,
        max_iter=cfg.kmeans_max_iter,
        random_state=cfg.kmeans_random_state,
    )
    labels = km.fit_predict(x)
    centers = km.cluster_centers_.astype(np.float32)
    return labels.astype(np.int32), centers


def _kmeans_opencv(x: np.ndarray, k: int, cfg: TeamClusteringConfig) -> Tuple[np.ndarray, np.ndarray]:
    import cv2

    data = x.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(cfg.kmeans_max_iter), 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    _, labels, centers = cv2.kmeans(data, k, None, criteria, int(max(1, cfg.kmeans_n_init)), flags)
    return labels.reshape(-1).astype(np.int32), centers.astype(np.float32)


def cluster_colors(colors: np.ndarray, cfg: TeamClusteringConfig | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster Nx3 color vectors into 2 teams using KMeans.

    Returns:
      labels: (N,) int32 in {0, 1}
      centers: (2, 3) float32
    """
    if cfg is None:
        cfg = TeamClusteringConfig()

    x = np.asarray(colors, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"colors must be Nx3, got shape {x.shape}")

    n = x.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((cfg.global_k, 3), dtype=np.float32)

    if n == 1:
        centers = np.vstack([x[0], x[0]]).astype(np.float32)
        return np.array([0], dtype=np.int32), centers

    k = int(cfg.global_k)
    try:
        labels, centers = _kmeans_sklearn(x, k, cfg)
    except Exception as e:
        logger.info("sklearn KMeans unavailable/failed (%s); falling back to OpenCV kmeans", e)
        labels, centers = _kmeans_opencv(x, k, cfg)

    return labels, centers

