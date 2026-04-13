"""
Thin session wrapper for HTTP/gRPC handlers: one instance per uploaded video / stream.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import supervision as sv

from .pipeline import FrameTrackingPipeline
from .schemas import TrackingConfig


class TrackingSession:
    """
    Holds a ``FrameTrackingPipeline`` with explicit lifecycle for API use.

    - Call ``reset()`` when starting a new video or after an error.
    - Call ``process_frame`` each frame; returns a JSON-serializable dict.
    """

    def __init__(self, config: TrackingConfig | None = None):
        self.pipeline = FrameTrackingPipeline(config or TrackingConfig())

    def reset(self) -> None:
        self.pipeline.reset()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        detections: sv.Detections,
        *,
        frame_index: int = 0,
    ) -> dict[str, Any]:
        _dets, result = self.pipeline.step(frame_bgr, detections, frame_index=frame_index)
        return result.to_dict()

    def process_frame_full(
        self,
        frame_bgr: np.ndarray,
        detections: sv.Detections,
        *,
        frame_index: int = 0,
    ) -> tuple[sv.Detections, dict[str, Any]]:
        """Returns supervision detections (updated) plus the JSON payload."""
        dets, result = self.pipeline.step(frame_bgr, detections, frame_index=frame_index)
        return dets, result.to_dict()
