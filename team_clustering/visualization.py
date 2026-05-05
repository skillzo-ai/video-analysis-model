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

    if cfg.draw_text and player_id is not None:
        # Centered above the player; text and outline use team color for readability on court.
        label = f"ID {int(player_id)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = float(cfg.text_scale)
        thickness = max(1, int(cfg.text_thickness))
        (tw, th), bl = cv2.getTextSize(label, font, font_scale, thickness)
        pad = 3
        baseline_y = max(th + pad, y - pad)
        tx_left = int(round(cx - tw * 0.5))
        tx_left = max(0, min(tx_left, w_img - tw - 2 * pad))
        bg_x1, bg_y1 = tx_left - pad, baseline_y - th - pad
        bg_x2, bg_y2 = tx_left + tw + pad, baseline_y + bl + pad
        bg_x1, bg_y1 = max(0, bg_x1), max(0, bg_y1)
        bg_x2, bg_y2 = min(w_img - 1, bg_x2), min(h_img - 1, bg_y2)
        cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1, cv2.LINE_AA)
        cv2.putText(
            out,
            label,
            (tx_left, baseline_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    elif cfg.draw_text:
        label = str(team_label)
        tx = x
        baseline_y = max(12, y - 4)
        cv2.putText(
            out,
            label,
            (tx, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(cfg.text_scale),
            color,
            int(cfg.text_thickness),
            cv2.LINE_AA,
        )

    return out

