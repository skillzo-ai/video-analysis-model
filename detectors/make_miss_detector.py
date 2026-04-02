from __future__ import annotations

from config.thresholds import EVENT_COOLDOWN_FRAMES
from core.geometry import point_inside_bbox
from models.data_structures import BallState, Hoop


class MakeMissDetector:
    """
    Detects when the ball center enters the hoop bounding box (treated as a make).
    Uses an edge trigger (outside -> inside) plus cooldown to limit repeats.
    """

    def __init__(self, cooldown_frames: int = EVENT_COOLDOWN_FRAMES):
        self.cooldown_frames = int(cooldown_frames)
        self._was_inside: bool = False
        self._cooldown_remaining: int = 0

    def reset(self) -> None:
        self._was_inside = False
        self._cooldown_remaining = 0

    def detect(self, ball: BallState, hoop: Hoop | None) -> bool:
        can_fire = self._cooldown_remaining == 0
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if hoop is None:
            self._was_inside = False
            return False

        inside = point_inside_bbox(ball.position, hoop.bbox)
        fired = False
        if inside and not self._was_inside and can_fire:
            fired = True
            self._cooldown_remaining = self.cooldown_frames

        self._was_inside = inside
        return fired
