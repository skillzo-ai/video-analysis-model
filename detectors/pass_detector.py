from __future__ import annotations

from config.thresholds import EVENT_COOLDOWN_FRAMES, PASS_DISTANCE_THRESHOLD
from core.geometry import euclidean_distance
from models.data_structures import BallState, PlayerState


class PassDetector:
    """
    Flags a pass when ball ownership changes and the new owner is farther than
    a pixel threshold from the previous owner's last known position.
    """

    def __init__(
        self,
        distance_threshold: float = PASS_DISTANCE_THRESHOLD,
        cooldown_frames: int = EVENT_COOLDOWN_FRAMES,
    ):
        self.distance_threshold = float(distance_threshold)
        self.cooldown_frames = int(cooldown_frames)
        self._prev_owner_id: int | None = None
        self._prev_owner_pos: tuple[float, float] | None = None
        self._cooldown_remaining: int = 0

    def reset(self) -> None:
        self._prev_owner_id = None
        self._prev_owner_pos = None
        self._cooldown_remaining = 0

    def detect(
        self,
        ball: BallState,
        players: list[PlayerState],
        owner_id: int | None,
    ) -> bool:
        """
        `owner_id` should match closest-player logic (e.g. from assign_ball_owner).
        """
        can_fire = self._cooldown_remaining == 0
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        pos_by_id = {int(p.player_id): p.position for p in players}
        fired = False

        if owner_id is not None:
            current_pos = pos_by_id.get(int(owner_id))

            if (
                self._prev_owner_id is not None
                and owner_id != self._prev_owner_id
                and current_pos is not None
                and self._prev_owner_pos is not None
            ):
                dist = euclidean_distance(self._prev_owner_pos, current_pos)
                if dist > self.distance_threshold and can_fire:
                    fired = True
                    self._cooldown_remaining = self.cooldown_frames

            if current_pos is not None:
                self._prev_owner_pos = current_pos
            self._prev_owner_id = int(owner_id)
        else:
            self._prev_owner_id = None

        return fired
