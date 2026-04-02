from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .clustering import cluster_colors
from .color_extraction import extract_color
from .config import TeamClusteringConfig
from .crop import crop_player
from .visualization import draw_player_ellipse

logger = logging.getLogger(__name__)


def _ensure_logger(cfg: TeamClusteringConfig) -> None:
    """
    Safe default logging setup for library usage.
    Does nothing if logging already configured by the host app.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    level = logging.DEBUG if cfg.debug_verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def classify_teams(
    frame: np.ndarray,
    bboxes: Sequence[Sequence[float | int]],
    cfg: TeamClusteringConfig | None = None,
) -> Dict[str, Any]:
    """
    Classify players into 2 teams using jersey color clustering.

    Args:
        frame: BGR image.
        bboxes: list of (x, y, w, h).
        cfg: configuration.

    Returns:
        {
          "labels": List[int],            # 0/1 aligned with input bboxes
          "teams": List[str],             # "Team A"/"Team B"
          "cluster_centers": np.ndarray,  # (2,3)
          "annotated_frame": np.ndarray   # BGR frame with ellipses
        }
    """
    if cfg is None:
        cfg = TeamClusteringConfig()
    _ensure_logger(cfg)

    if frame is None or frame.size == 0:
        return {"labels": [], "teams": [], "cluster_centers": np.zeros((2, 3), dtype=np.float32), "annotated_frame": frame}

    n = len(bboxes)
    if n == 0:
        return {"labels": [], "teams": [], "cluster_centers": np.zeros((2, 3), dtype=np.float32), "annotated_frame": frame.copy()}

    if cfg.debug:
        Path(cfg.debug_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: crop -> extract color
    colors: List[np.ndarray] = []
    valid_mask = np.zeros((n,), dtype=bool)
    for i, bbox in enumerate(bboxes):
        crop = crop_player(frame, bbox, cfg)
        if crop is None:
            colors.append(np.zeros((3,), dtype=np.float32))
            logger.debug("bbox[%d] invalid/empty crop, using zeros", i)
            continue
        valid_mask[i] = True
        vec = extract_color(crop, cfg, debug_prefix=f"p{i:03d}")
        colors.append(vec)

    color_arr = np.vstack(colors).astype(np.float32)  # (N,3)

    # Step 2: cluster (only valid players if possible)
    if valid_mask.any() and int(valid_mask.sum()) >= 2:
        labels_valid, centers = cluster_colors(color_arr[valid_mask], cfg)
        labels = np.full((n,), 0, dtype=np.int32)
        labels[valid_mask] = labels_valid
    else:
        # Degenerate case: 0-1 valid crop(s)
        labels = np.zeros((n,), dtype=np.int32)
        centers = np.zeros((2, 3), dtype=np.float32)
        if valid_mask.any():
            first = color_arr[valid_mask][0]
            centers[0] = first
            centers[1] = first

    if cfg.debug:
        logger.info("cluster_centers=%s", centers)

    # Step 3: map to Team A / Team B
    teams: List[str] = ["Team A" if int(l) == 0 else "Team B" for l in labels.tolist()]

    # Step 4: visualize
    annotated = annotate_frame(frame, bboxes, teams, cfg=cfg)

    return {
        "labels": labels.astype(int).tolist(),
        "teams": teams,
        "cluster_centers": centers,
        "annotated_frame": annotated,
    }


def classify_teams_no_draw(
    frame: np.ndarray,
    bboxes: Sequence[Sequence[float | int]],
    cfg: TeamClusteringConfig | None = None,
) -> Dict[str, Any]:
    """
    Same as classify_teams(), but does not draw anything.
    Useful for video pipelines where visualization is handled elsewhere.
    """
    if cfg is None:
        cfg = TeamClusteringConfig()
    _ensure_logger(cfg)

    if frame is None or frame.size == 0:
        return {"labels": [], "teams": [], "cluster_centers": np.zeros((2, 3), dtype=np.float32)}

    n = len(bboxes)
    if n == 0:
        return {"labels": [], "teams": [], "cluster_centers": np.zeros((2, 3), dtype=np.float32)}

    if cfg.debug:
        Path(cfg.debug_dir).mkdir(parents=True, exist_ok=True)

    colors: List[np.ndarray] = []
    valid_mask = np.zeros((n,), dtype=bool)
    for i, bbox in enumerate(bboxes):
        crop = crop_player(frame, bbox, cfg)
        if crop is None:
            colors.append(np.zeros((3,), dtype=np.float32))
            continue
        valid_mask[i] = True
        colors.append(extract_color(crop, cfg, debug_prefix=f"p{i:03d}"))

    color_arr = np.vstack(colors).astype(np.float32)

    if valid_mask.any() and int(valid_mask.sum()) >= 2:
        labels_valid, centers = cluster_colors(color_arr[valid_mask], cfg)
        labels = np.full((n,), 0, dtype=np.int32)
        labels[valid_mask] = labels_valid
    else:
        labels = np.zeros((n,), dtype=np.int32)
        centers = np.zeros((2, 3), dtype=np.float32)
        if valid_mask.any():
            first = color_arr[valid_mask][0]
            centers[0] = first
            centers[1] = first

    teams: List[str] = ["Team A" if int(l) == 0 else "Team B" for l in labels.tolist()]
    return {"labels": labels.astype(int).tolist(), "teams": teams, "cluster_centers": centers}


def annotate_frame(
    frame: np.ndarray,
    bboxes: Sequence[Sequence[float | int]],
    labels_or_teams: Sequence[int | str],
    cfg: TeamClusteringConfig | None = None,
) -> np.ndarray:
    """
    Draw team-specific ellipses under each player.
    labels_or_teams can be List[int] (0/1) or List[str] ("Team A"/"Team B").
    """
    if cfg is None:
        cfg = TeamClusteringConfig()
    _ensure_logger(cfg)

    if frame is None or frame.size == 0:
        return frame

    out = frame.copy()
    for i, (bbox, lab) in enumerate(zip(bboxes, labels_or_teams)):
        team = lab if isinstance(lab, str) else ("Team A" if int(lab) == 0 else "Team B")
        out = draw_player_ellipse(out, bbox, team, cfg, player_id=i if cfg.draw_text else None)
    return out

