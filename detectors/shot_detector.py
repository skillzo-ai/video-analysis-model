from __future__ import annotations

import math

from config.thresholds import (
    SHOT_COOLDOWN_FRAMES,
    SHOT_CONFIRM_FRAMES,
    SHOT_FLEX_MIN_COS_TO_RIM,
    SHOT_FLEX_MIN_DEPTH_BELOW_RIM_PX,
    SHOT_FLEX_MIN_SPEED,
    SHOT_FLEX_MIN_UPWARD_SPEED,
    SHOT_MIN_COS_TO_RIM,
    SHOT_MIN_SPEED,
    SHOT_MIN_UPWARD_SPEED,
    SHOT_RIM_TOP_FRACTION,
)
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


class ShotDetector:
    """
    Shot cue (two ways, either can arm the same confirm counter):

    1) Strict: clear upward speed, decent speed magnitude, velocity aimed at rim anchor.
    2) Flexible: milder upward motion, ball below the rim anchor (release zone), still
       loosely aimed at the rim — catches softer set shots / arc where strict cos fails.

    Rim anchor = top band of hoop bbox (not geometric center of the whole detection).
    """

    def __init__(
        self,
        min_upward_speed: float = SHOT_MIN_UPWARD_SPEED,
        min_cos_to_rim: float = SHOT_MIN_COS_TO_RIM,
        min_speed: float = SHOT_MIN_SPEED,
        flex_min_upward_speed: float = SHOT_FLEX_MIN_UPWARD_SPEED,
        flex_min_cos_to_rim: float = SHOT_FLEX_MIN_COS_TO_RIM,
        flex_min_speed: float = SHOT_FLEX_MIN_SPEED,
        flex_min_depth_below_rim_px: float = SHOT_FLEX_MIN_DEPTH_BELOW_RIM_PX,
        rim_top_fraction: float = SHOT_RIM_TOP_FRACTION,
        confirm_frames: int = SHOT_CONFIRM_FRAMES,
        cooldown_frames: int = SHOT_COOLDOWN_FRAMES,
    ):
        self.min_upward_speed = float(min_upward_speed)
        self.min_cos_to_rim = float(min_cos_to_rim)
        self.min_speed = float(min_speed)
        self.flex_min_upward_speed = float(flex_min_upward_speed)
        self.flex_min_cos_to_rim = float(flex_min_cos_to_rim)
        self.flex_min_speed = float(flex_min_speed)
        self.flex_min_depth_below_rim_px = float(flex_min_depth_below_rim_px)
        self.rim_top_fraction = float(rim_top_fraction)
        self.confirm_frames = max(1, int(confirm_frames))
        self.cooldown_frames = int(cooldown_frames)
        self._cooldown_remaining: int = 0
        self._confirm_streak: int = 0

    def reset(self) -> None:
        self._cooldown_remaining = 0
        self._confirm_streak = 0

    @staticmethod
    def rim_anchor(hoop: Hoop, top_fraction: float) -> tuple[float, float]:
        x1, y1, x2, y2 = _normalize_xyxy(hoop.bbox)
        cx = (x1 + x2) * 0.5
        h = max(1.0, y2 - y1)
        ry = y1 + h * float(top_fraction)
        return (cx, ry)

    @staticmethod
    def _cosine_and_speed(
        ball_xy: tuple[float, float],
        velocity_xy: tuple[float, float],
        target_xy: tuple[float, float],
    ) -> tuple[float, float]:
        bx, by = float(ball_xy[0]), float(ball_xy[1])
        vx, vy = float(velocity_xy[0]), float(velocity_xy[1])
        tx = float(target_xy[0]) - bx
        ty = float(target_xy[1]) - by
        v_norm = math.hypot(vx, vy)
        t_norm = math.hypot(tx, ty)
        if v_norm <= 1e-6 or t_norm <= 1e-6:
            return 0.0, v_norm
        cos = (vx * tx + vy * ty) / (v_norm * t_norm)
        return cos, v_norm

    def _geometry_strict(self, ball: BallState, rim: tuple[float, float]) -> bool:
        cos, speed = self._cosine_and_speed(ball.position, ball.velocity, rim)
        vy = float(ball.velocity[1])
        return (
            vy <= -self.min_upward_speed
            and cos >= self.min_cos_to_rim
            and speed >= self.min_speed
        )

    def _geometry_flex(self, ball: BallState, rim: tuple[float, float]) -> bool:
        cos, speed = self._cosine_and_speed(ball.position, ball.velocity, rim)
        vy = float(ball.velocity[1])
        _, rim_y = rim
        by = float(ball.position[1])
        below_rim = by >= rim_y + self.flex_min_depth_below_rim_px
        return (
            vy <= -self.flex_min_upward_speed
            and cos >= self.flex_min_cos_to_rim
            and speed >= self.flex_min_speed
            and below_rim
        )

    def detect(self, ball: BallState, hoop: Hoop | None) -> bool:
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._confirm_streak = 0
            return False

        if hoop is None:
            self._confirm_streak = 0
            return False

        rim = self.rim_anchor(hoop, self.rim_top_fraction)
        ok = self._geometry_strict(ball, rim) or self._geometry_flex(ball, rim)

        if not ok:
            self._confirm_streak = 0
            return False

        self._confirm_streak += 1
        if self._confirm_streak < self.confirm_frames:
            return False

        self._cooldown_remaining = self.cooldown_frames
        self._confirm_streak = 0
        return True
