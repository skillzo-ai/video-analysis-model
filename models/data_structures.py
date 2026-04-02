from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BallState:
    """Ball center position and velocity in image coordinates (y increases downward)."""

    position: tuple[float, float]
    velocity: tuple[float, float]


@dataclass(frozen=True)
class PlayerState:
    """Tracked player identity and center position (x, y)."""

    player_id: int
    position: tuple[float, float]


@dataclass(frozen=True)
class Hoop:
    """Axis-aligned bounding box for the rim / hoop region: x1, y1, x2, y2."""

    bbox: tuple[float, float, float, float]
