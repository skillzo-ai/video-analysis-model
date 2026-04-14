"""
Reusable tracking stack: ByteTrack, multi-cue player Re-ID, optional Kalman smoothing.
Ball boxes come from ``detection_pipeline.ball_tracker.BallTracker`` (YOLO + temporal filter + interpolate).

Typical service usage::

    from tracking import FrameTrackingPipeline, TrackingConfig

    pipe = FrameTrackingPipeline(TrackingConfig.from_dict(request.json["tracking"]))
    dets, result = pipe.step(frame_bgr, detections, frame_index=idx)
    return result.to_dict()
"""

from .pipeline import FrameTrackingPipeline
from .schemas import FrameTrackResult, TrackingConfig, frame_result_from_dict
from .service import TrackingSession

__all__ = [
    "FrameTrackingPipeline",
    "FrameTrackResult",
    "TrackingConfig",
    "TrackingSession",
    "frame_result_from_dict",
]
