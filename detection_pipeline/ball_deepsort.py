"""
Ball track: BoxMOT DeepOcSort (appearance + Kalman x,y,s,r per track).
Classic DeepSORT is not packaged in BoxMOT; DeepOcSort is the supported ReID + Kalman pipeline.
"""

from __future__ import annotations

import numpy as np
import torch

from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
from boxmot.utils import WEIGHTS


class BallDeepOcSortTracker:
    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        half: bool = False,
        reid_weights=None,
        det_thresh: float = 0.15,
        max_age: int = 45,
        min_hits: int = 1,
        iou_threshold: float = 0.15,
        embedding_off: bool = False,
    ):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        rw = reid_weights or (WEIGHTS / "osnet_x0_25_msmt17.pt")
        self.tracker = DeepOcSort(
            reid_weights=rw,
            device=device,
            half=half,
            det_thresh=det_thresh,
            max_age=max_age,
            min_hits=min_hits,
            max_obs=max_age + 10,
            iou_threshold=iou_threshold,
            embedding_off=embedding_off,
            cmc_off=True,
            aw_off=False,
        )
        self.last_size_wh: tuple[float, float] | None = None
        self.last_meas_center: tuple[float, float] | None = None
        self.prev_meas_center: tuple[float, float] | None = None
        self._saw_real_detection: bool = False

    @property
    def initialized(self) -> bool:
        return self._saw_real_detection

    def reset(self):
        self.tracker.active_tracks.clear()
        self.tracker.frame_count = 0
        self.last_size_wh = None
        self.last_meas_center = None
        self.prev_meas_center = None
        self._saw_real_detection = False

    def _bbox_from_output_rows(self, out: np.ndarray) -> np.ndarray | None:
        if out is None or out.size == 0:
            return None
        rows = out.reshape(-1, out.shape[-1])
        if rows.shape[0] > 1:
            i = int(np.argmax(rows[:, 4].astype(np.float64)))
            row = rows[i]
        else:
            row = rows[0]
        return np.asarray(row[:4], dtype=np.float32)

    def _bbox_from_active_kalman(self) -> np.ndarray | None:
        best_bb = None
        best_tsu = 10**9
        for trk in self.tracker.active_tracks:
            if trk.time_since_update > self.tracker.max_age:
                continue
            if self.tracker.frame_count > self.tracker.min_hits and trk.hits < self.tracker.min_hits:
                continue
            tsu = int(trk.time_since_update)
            if tsu < best_tsu:
                best_tsu = tsu
                st = trk.get_state()
                best_bb = np.asarray(st, dtype=np.float32).reshape(-1)[:4].copy()
        return best_bb

    def update(
        self,
        frame: np.ndarray,
        ball_xyxy: np.ndarray | None,
        conf: float,
    ) -> tuple[tuple[float, float] | None, np.ndarray | None]:
        """
        Returns ((cx, cy), xyxy) from DeepOcSort Kalman + association, or (None, None).
        """
        if ball_xyxy is not None:
            self._saw_real_detection = True
            dets = np.array(
                [
                    [
                        float(ball_xyxy[0]),
                        float(ball_xyxy[1]),
                        float(ball_xyxy[2]),
                        float(ball_xyxy[3]),
                        float(conf),
                        0.0,
                    ]
                ],
                dtype=np.float32,
            )
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        if not self._saw_real_detection:
            return None, None

        out = self.tracker.update(dets, frame)
        xyxy = self._bbox_from_output_rows(out)
        if xyxy is None:
            xyxy = self._bbox_from_active_kalman()
        if xyxy is None:
            return None, None

        cx = float((xyxy[0] + xyxy[2]) * 0.5)
        cy = float((xyxy[1] + xyxy[3]) * 0.5)
        self.last_size_wh = (
            max(1.0, float(xyxy[2] - xyxy[0])),
            max(1.0, float(xyxy[3] - xyxy[1])),
        )

        if ball_xyxy is not None:
            mcx = float((ball_xyxy[0] + ball_xyxy[2]) * 0.5)
            mcy = float((ball_xyxy[1] + ball_xyxy[3]) * 0.5)
            if self.last_meas_center is not None:
                self.prev_meas_center = self.last_meas_center
            self.last_meas_center = (mcx, mcy)

        return (cx, cy), xyxy

    def center_to_bbox_xyxy(
        self,
        center_xy: tuple[float, float],
        default_wh: tuple[float, float] = (16.0, 16.0),
    ) -> np.ndarray:
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
