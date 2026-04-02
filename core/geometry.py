from __future__ import annotations

import math
from typing import Iterable


def euclidean_distance(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    return math.hypot(ax - bx, ay - by)


def point_inside_bbox(
    point: tuple[float, float],
    bbox: tuple[float, float, float, float],
) -> bool:
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return x1 <= x <= x2 and y1 <= y <= y2


def ball_moving_toward_target(
    ball_xy: tuple[float, float],
    velocity_xy: tuple[float, float],
    target_xy: tuple[float, float],
) -> bool:
    """
    True if velocity aligns with the vector from ball to target (dot product > 0).
    Caller can combine with a minimum dot threshold for noise rejection.
    """
    tx = float(target_xy[0]) - float(ball_xy[0])
    ty = float(target_xy[1]) - float(ball_xy[1])
    vx, vy = float(velocity_xy[0]), float(velocity_xy[1])
    return (vx * tx + vy * ty) > 0.0
