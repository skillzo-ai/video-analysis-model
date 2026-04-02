from __future__ import annotations

import math

from config.thresholds import (
    EVENT_COOLDOWN_FRAMES,
    SHOT_HOOP_DOT_MIN,
    SHOT_UPWARD_VELOCITY_THRESHOLD,
)
from core.geometry import ball_moving_toward_target
from models.data_structures import BallState, Hoop


class ShotDetector:
    """
    Heuristic shot cue: ball moving upward (image coordinates) and toward hoop center.
    """

    def __init__(
        self,
        upward_velocity_threshold: float = SHOT_UPWARD_VELOCITY_THRESHOLD,
        hoop_dot_min: float = SHOT_HOOP_DOT_MIN,
        cooldown_frames: int = EVENT_COOLDOWN_FRAMES,
    ):
        self.upward_velocity_threshold = float(upward_velocity_threshold)
        self.hoop_dot_min = float(hoop_dot_min)
        self.cooldown_frames = int(cooldown_frames)
        self._cooldown_remaining: int = 0

    def reset(self) -> None:
        self._cooldown_remaining = 0

    @staticmethod
    def _hoop_center(hoop: Hoop) -> tuple[float, float]:
        x1, y1, x2, y2 = hoop.bbox
        return ((float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5)

    @staticmethod
    def _velocity_toward_dot(
        ball_xy: tuple[float, float],
        velocity_xy: tuple[float, float],
        target_xy: tuple[float, float],
    ) -> float:
        tx = float(target_xy[0]) - float(ball_xy[0])
        ty = float(target_xy[1]) - float(ball_xy[1])
        vx, vy = float(velocity_xy[0]), float(velocity_xy[1])
        dot = vx * tx + vy * ty
        v_norm = math.hypot(vx, vy)
        t_norm = math.hypot(tx, ty)
        if v_norm <= 1e-6 or t_norm <= 1e-6:
            return 0.0
        return dot / (v_norm * t_norm)

    def detect(self, ball: BallState, hoop: Hoop | None) -> bool:
        can_fire = self._cooldown_remaining == 0
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if hoop is None:
            return False

        vx, vy = float(ball.velocity[0]), float(ball.velocity[1])
        # Upward in image coords => decreasing y.
        upward = vy <= -self.upward_velocity_threshold
        if not upward:
            return False

        center = self._hoop_center(hoop)
        if not ball_moving_toward_target(ball.position, ball.velocity, center):
            return False

        dot_norm = self._velocity_toward_dot(ball.position, ball.velocity, center)
        if dot_norm < self.hoop_dot_min:
            return False

        if not can_fire:
            return False

        self._cooldown_remaining = self.cooldown_frames
        return True
