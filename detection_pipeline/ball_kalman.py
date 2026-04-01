import numpy as np


class BallKalmanTracker:
    """
    Lightweight constant-velocity Kalman filter for 2D ball center:
    state = [x, y, vx, vy]
    measurement = [x, y]
    """

    def __init__(
        self,
        process_var: float = 25.0,
        meas_var: float = 100.0,
        init_var: float = 1e4,
    ):
        self.process_var = float(process_var)
        self.meas_var = float(meas_var)
        self.init_var = float(init_var)

        self._x = None  # (4, 1)
        self._P = None  # (4, 4)

        self.last_size_wh = None  # (w, h) from last observed bbox
        self.last_center = None  # (x, y)

    @property
    def initialized(self) -> bool:
        return self._x is not None

    def reset(self):
        self._x = None
        self._P = None
        self.last_size_wh = None
        self.last_center = None

    def _init_from_measurement(self, z_xy: np.ndarray):
        x, y = float(z_xy[0]), float(z_xy[1])
        self._x = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self._P = np.eye(4, dtype=np.float32) * self.init_var
        self.last_center = (x, y)

    def predict(self, dt: float = 1.0) -> tuple[float, float]:
        dt = float(dt)
        if not self.initialized:
            return (None, None)

        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        q = self.process_var
        Q = np.array(
            [
                [0.25 * dt**4, 0.0, 0.5 * dt**3, 0.0],
                [0.0, 0.25 * dt**4, 0.0, 0.5 * dt**3],
                [0.5 * dt**3, 0.0, dt**2, 0.0],
                [0.0, 0.5 * dt**3, 0.0, dt**2],
            ],
            dtype=np.float32,
        ) * q

        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

        x, y = float(self._x[0, 0]), float(self._x[1, 0])
        self.last_center = (x, y)
        return (x, y)

    def update(self, z_xy: tuple[float, float] | np.ndarray) -> tuple[float, float]:
        z = np.asarray(z_xy, dtype=np.float32).reshape(2)
        if not self.initialized:
            self._init_from_measurement(z)
            return (float(z[0]), float(z[1]))

        # Predict already done? This filter can be used with predict()->update() or update() alone.
        # We'll do a predict step with dt=1 inside update if caller didn't.
        self.predict(dt=1.0)

        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        R = np.eye(2, dtype=np.float32) * self.meas_var

        z = z.reshape(2, 1)
        y = z - (H @ self._x)
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)

        self._x = self._x + (K @ y)
        I = np.eye(4, dtype=np.float32)
        self._P = (I - (K @ H)) @ self._P

        x, y = float(self._x[0, 0]), float(self._x[1, 0])
        self.last_center = (x, y)
        return (x, y)

    def center_to_bbox_xyxy(self, center_xy: tuple[float, float], default_wh: tuple[float, float] = (16.0, 16.0)):
        cx, cy = float(center_xy[0]), float(center_xy[1])
        if self.last_size_wh is not None:
            w, h = self.last_size_wh
        else:
            w, h = float(default_wh[0]), float(default_wh[1])

        x1 = cx - (w / 2.0)
        y1 = cy - (h / 2.0)
        x2 = cx + (w / 2.0)
        y2 = cy + (h / 2.0)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

