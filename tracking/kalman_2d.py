"""
Constant-velocity Kalman filter per track id for smoothing (x, y) centers in image space.
"""

from __future__ import annotations

import numpy as np


class KalmanCenter2D:
    """
    4-state CV model: [cx, cy, vx, vy].
    Measurement: (cx, cy).
    """

    def __init__(
        self,
        cx: float,
        cy: float,
        *,
        meas_noise: float = 4.0,
        process_noise: float = 0.08,
    ):
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 10.0
        self.Q = np.eye(4, dtype=np.float64) * float(process_noise)
        r = float(meas_noise)
        self.R = np.eye(2, dtype=np.float64) * r
        self._F = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        self._H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)

    def predict(self) -> None:
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self.Q

    def update(self, zx: float, zy: float) -> tuple[float, float]:
        z = np.array([zx, zy], dtype=np.float64)
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self.R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self._H) @ self.P
        return float(self.x[0]), float(self.x[1])

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])


class PerIdKalmanBank:
    """One Kalman filter per global player id."""

    def __init__(self, meas_noise: float = 4.0, process_noise: float = 0.08):
        self._meas_noise = meas_noise
        self._process_noise = process_noise
        self._filters: dict[int, KalmanCenter2D] = {}

    def reset(self) -> None:
        self._filters.clear()

    def smooth_center(self, global_id: int, cx: float, cy: float) -> tuple[float, float]:
        gid = int(global_id)
        kf = self._filters.get(gid)
        if kf is None:
            kf = KalmanCenter2D(
                cx,
                cy,
                meas_noise=self._meas_noise,
                process_noise=self._process_noise,
            )
            self._filters[gid] = kf
            return cx, cy
        kf.predict()
        return kf.update(cx, cy)
