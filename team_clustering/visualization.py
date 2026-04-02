from __future__ import annotations

import logging
from typing import Sequence, Tuple

import cv2
import numpy as np

from .config import TeamClusteringConfig

logger = logging.getLogger(__name__)


def _clip_bbox_xywh(bbox: Sequence[float | int], w_img: int, h_img: int) -> Tuple[int, int, int, int] | None:
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


def draw_player_ellipse(
    frame: np.ndarray,
    bbox: Sequence[float | int],
    team_label: str,
    cfg: TeamClusteringConfig | None = None,
    *,
    player_id: int | None = None,
) -> np.ndarray:
    """
    Draw a team-colored ellipse at the player's feet (bottom of bbox).

    Requirements:
    - Center: (x + w/2, y + h)
    - Axes: (w/2, h*0.15) (scaled by config)
    - Team A → Blue (BGR 255,0,0)
    - Team B → Red  (BGR 0,0,255)
    - Thickness 2–3 px
    - Anti-aliased: cv2.LINE_AA
    """
    if cfg is None:
        cfg = TeamClusteringConfig()

    if frame is None or frame.size == 0:
        return frame

    h_img, w_img = frame.shape[:2]
    clipped = _clip_bbox_xywh(bbox, w_img=w_img, h_img=h_img)
    if clipped is None:
        return frame

    x, y, w, h = clipped
    cx = int(round(x + w * 0.5))
    cy = int(round(y + h))

    ax = max(2, int(round(w * float(cfg.ellipse_axis_w_scale))))
    ay = max(2, int(round(h * float(cfg.ellipse_axis_h_scale))))

    if team_label == "Team A":
        color = cfg.ellipse_color_team_a_bgr
    else:
        color = cfg.ellipse_color_team_b_bgr

    out = frame

    # Ellipse (feet marker)
    cv2.ellipse(
        out,
        center=(cx, cy),
        axes=(ax, ay),
        angle=0.0,
        startAngle=0.0,
        endAngle=360.0,
        color=color,
        thickness=int(cfg.ellipse_thickness),
        lineType=cv2.LINE_AA,
    )

    if cfg.draw_bbox:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

    if cfg.draw_text:
        label = team_label if player_id is None else f"{team_label} #{player_id}"
        tx = x
        ty = max(0, y - 5)
        cv2.putText(
            out,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(cfg.text_scale),
            color,
            int(cfg.text_thickness),
            cv2.LINE_AA,
        )

    return out

