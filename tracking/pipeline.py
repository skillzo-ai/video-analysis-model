"""
Composable frame step: ByteTrack -> identity bridge -> optional Kalman smoothing.
Designed for service/API use: pass ``FrameTrackResult.to_dict()`` as JSON.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

from .identity_bridge import PlayerIdentityBridge
from .kalman_2d import PerIdKalmanBank
from .schemas import FrameTrackResult, PlayerTrackRecord, TrackingConfig


class FrameTrackingPipeline:
    """
    Run after a detector produces ``sv.Detections`` (no tracker_id yet).

    Order:
      1. ByteTrack (supervision) assigns ``tracker_id`` per object.
      2. ``PlayerIdentityBridge`` remaps player ``tracker_id`` to stable global ids
         (visual + optional jersey color + height).
      3. Optional per-id Kalman smoothing of player bbox centers.
    """

    def __init__(
        self,
        config: TrackingConfig | None = None,
        *,
        byte_tracker: sv.ByteTrack | None = None,
        identity: PlayerIdentityBridge | None = None,
        kalman: PerIdKalmanBank | None = None,
    ):
        self.config = config or TrackingConfig()
        self.tracker = byte_tracker or sv.ByteTrack(
            track_activation_threshold=self.config.track_activation_threshold,
            lost_track_buffer=self.config.lost_track_buffer,
            minimum_matching_threshold=self.config.minimum_matching_threshold,
        )
        self.identity = identity or PlayerIdentityBridge(self.config)
        self.kalman = kalman or PerIdKalmanBank(
            meas_noise=self.config.kalman_measurement_noise,
            process_noise=self.config.kalman_process_noise,
        )

    def reset(self) -> None:
        """Clear identity memory and Kalman filters (new video / session)."""
        self.identity.reset()
        self.kalman.reset()

    def step(
        self,
        frame_bgr: np.ndarray,
        detections: sv.Detections,
        frame_index: int = 0,
    ) -> tuple[sv.Detections, FrameTrackResult]:
        """
        Returns updated detections (with stable player ids) and a JSON-serializable result.
        """
        dets = self.tracker.update_with_detections(detections)
        dets, cues_per_row = self.identity.update(frame_bgr, dets)

        if self.config.use_kalman_smooth:
            dets = self._apply_kalman_xyxy(frame_bgr, dets)

        fr = self._build_frame_result(frame_bgr, dets, frame_index, cues_per_row)
        return dets, fr

    def _apply_kalman_xyxy(self, frame_bgr: np.ndarray, dets: sv.Detections) -> sv.Detections:
        cls = dets.class_id
        tid = dets.tracker_id
        if cls is None or tid is None:
            return dets
        xyxy = dets.xyxy.copy()
        h_frame = float(frame_bgr.shape[0])
        pid = int(self.config.player_class_id)
        for i in range(len(dets)):
            if int(cls[i]) != pid:
                continue
            x1, y1, x2, y2 = xyxy[i].astype(np.float32)
            w = max(1.0, float(x2 - x1))
            h = max(1.0, float(y2 - y1))
            cx = float((x1 + x2) * 0.5)
            cy = float((y1 + y2) * 0.5)
            gid = int(tid[i])
            scx, scy = self.kalman.smooth_center(gid, cx, cy)
            nx1 = scx - w * 0.5
            ny1 = scy - h * 0.5
            nx2 = scx + w * 0.5
            ny2 = scy + h * 0.5
            # Clamp to frame
            fh, fw = frame_bgr.shape[:2]
            nx1 = float(np.clip(nx1, 0, fw - 1))
            nx2 = float(np.clip(nx2, 0, fw - 1))
            ny1 = float(np.clip(ny1, 0, fh - 1))
            ny2 = float(np.clip(ny2, 0, fh - 1))
            xyxy[i] = [nx1, ny1, nx2, ny2]
        out = dets
        out.xyxy = xyxy
        return out

    def _build_frame_result(
        self,
        frame_bgr: np.ndarray,
        dets: sv.Detections,
        frame_index: int,
        cues_per_row: list[dict[str, object] | None] | None = None,
    ) -> FrameTrackResult:
        cls = dets.class_id
        tid = dets.tracker_id
        conf = dets.confidence
        players: list[PlayerTrackRecord] = []
        identity_map: dict[str, int] = {}
        cues_per_row = cues_per_row or [None] * len(dets)
        pid = int(self.config.player_class_id)
        if cls is None or tid is None:
            return FrameTrackResult(
                frame_index=frame_index,
                players=[],
                identity_map={},
                extras={"frame_shape": list(frame_bgr.shape[:2])},
            )

        for i in range(len(dets)):
            if int(cls[i]) != pid:
                continue
            bb = dets.xyxy[i].astype(np.float32)
            g = int(tid[i])
            identity_map[str(i)] = g
            cx = float((bb[0] + bb[2]) * 0.5)
            cy = float((bb[1] + bb[3]) * 0.5)
            cf = float(conf[i]) if conf is not None else None
            raw_cues = cues_per_row[i] if i < len(cues_per_row) else None
            cue_out: dict[str, object] = {}
            if isinstance(raw_cues, dict):
                for k, v in raw_cues.items():
                    if k != "global_id":
                        cue_out[k] = v
            players.append(
                PlayerTrackRecord(
                    global_id=g,
                    tracker_id=g,
                    xyxy=(float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])),
                    confidence=cf,
                    center_xy=(cx, cy),
                    cues_used=cue_out,
                )
            )

        return FrameTrackResult(
            frame_index=frame_index,
            players=players,
            identity_map=identity_map,
            extras={
                "frame_shape": list(frame_bgr.shape[:2]),
                "algorithms": ["ByteTrack", "appearance_reid", "jersey_color_optional", "kalman_smooth_optional"],
            },
        )
