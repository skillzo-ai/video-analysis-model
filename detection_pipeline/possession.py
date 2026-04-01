import numpy as np


class PossessionAssigner:
    """
    Assigns ball possession to nearest player (by center distance),
    with simple hysteresis to avoid flicker.
    """

    def __init__(self, max_dist_px: float = 120.0, switch_confirm_frames: int = 3, keep_frames_when_lost: int = 10):
        self.max_dist_px = float(max_dist_px)
        self.switch_confirm_frames = int(switch_confirm_frames)
        self.keep_frames_when_lost = int(keep_frames_when_lost)

        self.current_player_id = None
        self._candidate_id = None
        self._candidate_count = 0
        self._lost_count = 0

    def reset(self):
        self.current_player_id = None
        self._candidate_id = None
        self._candidate_count = 0
        self._lost_count = 0

    @staticmethod
    def _centers_from_xyxy(xyxy: np.ndarray) -> np.ndarray:
        # xyxy: (N, 4) -> centers: (N, 2)
        c = np.empty((xyxy.shape[0], 2), dtype=np.float32)
        c[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
        c[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
        return c

    def update(self, ball_center_xy: tuple[float, float] | None, player_xyxy: np.ndarray, player_ids: np.ndarray):
        """
        Returns: current_player_id (or None)
        """
        if ball_center_xy is None or player_xyxy is None or len(player_xyxy) == 0:
            self._lost_count += 1
            if self._lost_count > self.keep_frames_when_lost:
                self.current_player_id = None
            return self.current_player_id

        self._lost_count = 0

        centers = self._centers_from_xyxy(player_xyxy.astype(np.float32))
        ball = np.array([[float(ball_center_xy[0]), float(ball_center_xy[1])]], dtype=np.float32)
        dists = np.linalg.norm(centers - ball, axis=1)

        nearest_idx = int(np.argmin(dists))
        nearest_dist = float(dists[nearest_idx])
        nearest_id = int(player_ids[nearest_idx])

        if nearest_dist > self.max_dist_px:
            self._lost_count += 1
            if self._lost_count > self.keep_frames_when_lost:
                self.current_player_id = None
            return self.current_player_id

        # If already holding and same player stays closest, keep.
        if self.current_player_id == nearest_id:
            self._candidate_id = None
            self._candidate_count = 0
            return self.current_player_id

        # Candidate for switching
        if self._candidate_id != nearest_id:
            self._candidate_id = nearest_id
            self._candidate_count = 1
        else:
            self._candidate_count += 1

        if self._candidate_count >= self.switch_confirm_frames:
            self.current_player_id = nearest_id
            self._candidate_id = None
            self._candidate_count = 0

        return self.current_player_id

