"""
JSON-friendly tracking configuration and per-frame results for API / service use.

All types are plain dataclasses with ``to_dict`` / ``from_dict`` for REST or message payloads
without requiring Pydantic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class TrackingConfig:
    """
    Tunables for multi-object tracking aligned with ByteTrack + Re-ID + Kalman smoothing.

    Deep appearance association for the ball is handled separately in ``BallDeepOcSortTracker``;
    for players, persistent identity across occlusions / exit-reentry uses visual Re-ID plus
    optional jersey color and bbox-height cues.
    """

    # ByteTrack (supervision)
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 150
    minimum_matching_threshold: float = 0.8

    # Global identity bridge (after ByteTrack assigns tracker_id)
    reid_similarity_threshold: float = 0.85
    max_embedding_history: int = 12
    embedding_compare_last: int = 5

    # Multi-cue fusion weights (should sum to ~1.0 for interpretability)
    use_jersey_color: bool = True
    use_height_cue: bool = True
    embedding_weight: float = 0.72
    jersey_color_weight: float = 0.18
    height_weight: float = 0.10

    # Kalman smoothing on per-global-id centers (players)
    use_kalman_smooth: bool = True
    kalman_measurement_noise: float = 4.0
    kalman_process_noise: float = 0.08

    # Class ids (YOLO / custom models)
    player_class_id: int = 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrackingConfig:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


@dataclass
class PlayerTrackRecord:
    """One player row after tracking + identity + optional smoothing."""

    global_id: int
    tracker_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float | None
    center_xy: tuple[float, float]
    cues_used: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameTrackResult:
    """Serializable snapshot for a single frame (e.g. API response body)."""

    frame_index: int
    players: list[PlayerTrackRecord]
    identity_map: dict[str, int]  # str(track_id) -> global_id for JSON keys
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "players": [
                {
                    "global_id": p.global_id,
                    "tracker_id": p.tracker_id,
                    "xyxy": list(p.xyxy),
                    "confidence": p.confidence,
                    "center_xy": list(p.center_xy),
                    "cues_used": p.cues_used,
                }
                for p in self.players
            ],
            "identity_map": self.identity_map,
            "extras": self.extras,
        }


def frame_result_from_dict(d: dict[str, Any]) -> FrameTrackResult:
    players_raw = d.get("players") or []
    players: list[PlayerTrackRecord] = []
    for p in players_raw:
        xy = p.get("xyxy") or [0, 0, 0, 0]
        cxy = p.get("center_xy") or [0.0, 0.0]
        players.append(
            PlayerTrackRecord(
                global_id=int(p["global_id"]),
                tracker_id=int(p["tracker_id"]),
                xyxy=(float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])),
                confidence=p.get("confidence"),
                center_xy=(float(cxy[0]), float(cxy[1])),
                cues_used=dict(p.get("cues_used") or {}),
            )
        )
    imap = d.get("identity_map") or {}
    identity_map = {str(k): int(v) for k, v in imap.items()}
    return FrameTrackResult(
        frame_index=int(d.get("frame_index", 0)),
        players=players,
        identity_map=identity_map,
        extras=dict(d.get("extras") or {}),
    )
