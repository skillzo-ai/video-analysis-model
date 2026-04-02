from __future__ import annotations

from config.thresholds import (
    MAKE_APPROACH_MAX_DIST_DELTA_PX,
    MAKE_COOLDOWN_FRAMES,
    MAKE_OPENING_HEIGHT_FRAC,
    MAKE_OPENING_TOP_PAD_FRAC,
    MAKE_OPENING_WIDTH_FRAC,
    MAKE_REQUIRE_APPROACH,
)
from core.geometry import point_inside_bbox
from models.data_structures import BallState, Hoop


def _normalize_xyxy(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


class MakeMissDetector:
    """
    Make = ball center enters a tight *rim opening* ROI (top-centered slice of the hoop
    bbox), not the full detector box — avoids counting while the ball is still below
    the backboard or grazing the outer box.

    Optional approach check: distance to rim anchor must not increase much vs last
    frame (ball generally closing on the hoop, not bouncing away).
    """

    def __init__(
        self,
        opening_width_frac: float = MAKE_OPENING_WIDTH_FRAC,
        opening_height_frac: float = MAKE_OPENING_HEIGHT_FRAC,
        opening_top_pad_frac: float = MAKE_OPENING_TOP_PAD_FRAC,
        require_approach: bool = MAKE_REQUIRE_APPROACH,
        approach_max_dist_delta_px: float = MAKE_APPROACH_MAX_DIST_DELTA_PX,
        rim_anchor_top_fraction: float = 0.22,
        cooldown_frames: int = MAKE_COOLDOWN_FRAMES,
    ):
        self.opening_width_frac = float(opening_width_frac)
        self.opening_height_frac = float(opening_height_frac)
        self.opening_top_pad_frac = float(opening_top_pad_frac)
        self.require_approach = bool(require_approach)
        self.approach_max_dist_delta_px = float(approach_max_dist_delta_px)
        self.rim_anchor_top_fraction = float(rim_anchor_top_fraction)
        self.cooldown_frames = int(cooldown_frames)

        self._was_inside_opening: bool = False
        self._cooldown_remaining: int = 0
        self._prev_dist_to_rim: float | None = None

    def reset(self) -> None:
        self._was_inside_opening = False
        self._cooldown_remaining = 0
        self._prev_dist_to_rim = None

    @staticmethod
    def _opening_bbox(hoop: Hoop, w_frac: float, h_frac: float, top_pad_frac: float) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = _normalize_xyxy(hoop.bbox)
        w = x2 - x1
        h = y2 - y1
        ow = max(4.0, w * w_frac)
        oh = max(4.0, h * h_frac)
        cx = (x1 + x2) * 0.5
        top = y1 + h * top_pad_frac
        ox1 = cx - ow * 0.5
        oy1 = top
        ox2 = cx + ow * 0.5
        oy2 = top + oh
        return (ox1, oy1, ox2, oy2)

    @staticmethod
    def _rim_anchor(hoop: Hoop, top_fraction: float) -> tuple[float, float]:
        x1, y1, x2, y2 = _normalize_xyxy(hoop.bbox)
        cx = (x1 + x2) * 0.5
        hh = max(1.0, y2 - y1)
        ry = y1 + hh * float(top_fraction)
        return (cx, ry)

    @staticmethod
    def _dist(ax: float, ay: float, bx: float, by: float) -> float:
        dx, dy = ax - bx, ay - by
        return (dx * dx + dy * dy) ** 0.5

    def detect(self, ball: BallState, hoop: Hoop | None) -> bool:
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if hoop is None:
            self._was_inside_opening = False
            self._prev_dist_to_rim = None
            return False

        opening = self._opening_bbox(
            hoop,
            self.opening_width_frac,
            self.opening_height_frac,
            self.opening_top_pad_frac,
        )
        rim = self._rim_anchor(hoop, self.rim_anchor_top_fraction)
        bx, by = float(ball.position[0]), float(ball.position[1])
        dist_now = self._dist(bx, by, rim[0], rim[1])

        inside = point_inside_bbox(ball.position, opening)

        approach_ok = True
        if self.require_approach and self._prev_dist_to_rim is not None:
            # Allow small noise; reject clear retreat from rim.
            if dist_now > self._prev_dist_to_rim + self.approach_max_dist_delta_px:
                approach_ok = False

        self._prev_dist_to_rim = dist_now

        fired = False
        can_fire = self._cooldown_remaining == 0
        if (
            inside
            and not self._was_inside_opening
            and can_fire
            and approach_ok
        ):
            fired = True
            self._cooldown_remaining = self.cooldown_frames

        self._was_inside_opening = inside
        return fired
