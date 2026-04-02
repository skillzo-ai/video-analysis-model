from __future__ import annotations

from typing import Sequence

from .geometry import euclidean_distance


def assign_ball_owner(
    ball_xy: tuple[float, float],
    player_centers: Sequence[tuple[float, float]],
    player_ids: Sequence[int],
) -> tuple[int | None, float | None]:
    """
    Closest player to the ball by Euclidean distance in the image plane.

    Returns (owner_id, distance) or (None, None) if no players.
    """
    if not player_centers or not player_ids:
        return None, None
    if len(player_centers) != len(player_ids):
        raise ValueError("player_centers and player_ids must have the same length")

    best_id: int | None = None
    best_dist: float | None = None
    bx, by = float(ball_xy[0]), float(ball_xy[1])

    for (px, py), pid in zip(player_centers, player_ids):
        d = euclidean_distance((bx, by), (float(px), float(py)))
        if best_dist is None or d < best_dist:
            best_dist = d
            best_id = int(pid)

    return best_id, best_dist
