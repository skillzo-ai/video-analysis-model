"""
Ball-only detection + temporal cleanup (batch predict, wrong-detection filter, interpolation).

Aligned with the reference BallTracker workflow using ``ball_detector_model.pt`` and class name ``Ball``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


def read_stub(read_from_stub: bool, stub_path: str | None):
    if not read_from_stub or not stub_path:
        return None
    p = Path(stub_path)
    if not p.is_file():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def save_stub(stub_path: str | None, tracks) -> None:
    if stub_path is None:
        return
    p = Path(stub_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(tracks, f)


def _video_frame_count_fallback(source: str) -> int:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n


class BallTracker:
    """
    Basketball ball detection with batch inference, highest-confidence ``Ball`` pick per frame,
    movement-based filtering, and gap interpolation (pandas-free).
    """

    def __init__(
        self,
        model_path: str = "ball_detector_model.pt",
        *,
        batch_size: int = 20,
        conf: float = 0.5,
        ball_class_name: str = "Ball",
    ):
        self.model = YOLO(model_path)
        self.batch_size = int(batch_size)
        self.conf = float(conf)
        self.ball_class_name = ball_class_name
        self._ball_cls_id: int | None = None

    def _resolve_ball_class_id(self) -> int:
        if self._ball_cls_id is not None:
            return self._ball_cls_id
        names = getattr(self.model, "names", None) or {}
        target = self.ball_class_name.strip().lower()
        for k, v in names.items():
            if str(v).strip().lower() == target:
                self._ball_cls_id = int(k)
                return self._ball_cls_id
        # Fallback: single-class models usually use 0
        self._ball_cls_id = 0
        return self._ball_cls_id

    def detect_frames(self, frames: list):
        """Run batched ``predict`` (same as reference)."""
        detections: list = []
        bs = max(1, self.batch_size)
        for i in range(0, len(frames), bs):
            chunk = frames[i : i + bs]
            detections.extend(
                self.model.predict(chunk, conf=self.conf, verbose=False)
            )
        return detections

    def tracks_from_yolo_results(self, yolo_results: list) -> list[dict]:
        """One dict per frame: ``{1: {'bbox': [x1,y1,x2,y2]}}`` if a Ball was found."""
        ball_idx = self._resolve_ball_class_id()
        tracks: list[dict] = []
        for detection in yolo_results:
            ds = sv.Detections.from_ultralytics(detection)
            chosen_bbox = None
            max_confidence = 0.0
            if len(ds) > 0 and ds.class_id is not None:
                for i in range(len(ds)):
                    if int(ds.class_id[i]) != ball_idx:
                        continue
                    conf = float(ds.confidence[i]) if ds.confidence is not None else 0.0
                    if conf > max_confidence:
                        max_confidence = conf
                        chosen_bbox = ds.xyxy[i].astype(float).tolist()
            row: dict = {}
            if chosen_bbox is not None:
                row[1] = {"bbox": chosen_bbox}
            tracks.append(row)
        return tracks

    def remove_wrong_detections(self, ball_positions: list[dict]) -> list[dict]:
        """Drop detections whose top-left jumps farther than allowed vs last good frame."""
        maximum_allowed_distance = 25
        last_good_frame_index = -1
        out = [dict(x) for x in ball_positions]

        for i in range(len(out)):
            current_box = out[i].get(1, {}).get("bbox", [])

            if not isinstance(current_box, (list, tuple)) or len(current_box) != 4:
                continue

            if last_good_frame_index == -1:
                last_good_frame_index = i
                continue

            last_good_box = out[last_good_frame_index].get(1, {}).get("bbox", [])
            if not isinstance(last_good_box, (list, tuple)) or len(last_good_box) != 4:
                last_good_frame_index = i
                continue

            frame_gap = i - last_good_frame_index
            adjusted_max_distance = maximum_allowed_distance * max(1, frame_gap)

            d = float(
                np.linalg.norm(
                    np.array(last_good_box[:2], dtype=np.float64)
                    - np.array(current_box[:2], dtype=np.float64)
                )
            )
            if d > adjusted_max_distance:
                out[i] = {}
            else:
                last_good_frame_index = i

        return out

    def interpolate_ball_positions(self, ball_positions: list[dict]) -> list[dict]:
        """Linear interpolate + forward/backward fill (equivalent to pandas interpolate + bfill)."""
        rows: list[list[float]] = []
        for x in ball_positions:
            bb = x.get(1, {}).get("bbox", [])
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                rows.append([float(bb[j]) for j in range(4)])
            else:
                rows.append([np.nan, np.nan, np.nan, np.nan])

        arr = np.asarray(rows, dtype=np.float64)
        n = arr.shape[0]
        if n == 0:
            return ball_positions

        idx = np.arange(n, dtype=np.float64)
        for c in range(4):
            col = arr[:, c]
            ok = ~np.isnan(col)
            if not np.any(ok):
                continue
            if np.all(ok):
                continue
            col = col.copy()
            col[~ok] = np.interp(idx[~ok], idx[ok], col[ok])
            arr[:, c] = col

        # ffill
        for c in range(4):
            for i in range(1, n):
                if np.isnan(arr[i, c]):
                    arr[i, c] = arr[i - 1, c]
        # bfill
        for c in range(4):
            for i in range(n - 2, -1, -1):
                if np.isnan(arr[i, c]):
                    arr[i, c] = arr[i + 1, c]

        out: list[dict] = []
        for i in range(n):
            if np.any(np.isnan(arr[i])):
                out.append({})
            else:
                out.append({1: {"bbox": arr[i].tolist()}})
        return out

    def get_object_tracks(
        self,
        frames: list,
        read_from_stub: bool = False,
        stub_path: str | None = None,
    ) -> list[dict]:
        """
        Detect + filter + interpolate. Optional pickle cache at ``stub_path``.
        """
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None and len(tracks) == len(frames):
            return tracks

        detections = self.detect_frames(frames)
        tracks = self.tracks_from_yolo_results(detections)
        tracks = self.remove_wrong_detections(tracks)
        tracks = self.interpolate_ball_positions(tracks)
        save_stub(stub_path, tracks)
        return tracks

    def run_ball_pipeline_on_video(
        self,
        source: str,
        *,
        read_from_stub: bool = False,
        stub_path: str | None = None,
    ) -> list[dict]:
        """
        Full pass over ``source``: batched ball detection (streaming, no full-frame list),
        wrong-det removal, interpolation. Returns one dict per frame.
        """
        vi = sv.VideoInfo.from_video_path(source)
        total = getattr(vi, "total_frames", None)
        if total is None or total <= 0:
            total = _video_frame_count_fallback(source)

        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None and total > 0 and len(tracks) == total:
            return tracks

        yolo_results: list = []
        buffer: list = []
        for frame in sv.get_video_frames_generator(source):
            buffer.append(frame)
            if len(buffer) >= self.batch_size:
                yolo_results.extend(
                    self.model.predict(buffer, conf=self.conf, verbose=False)
                )
                buffer = []
        if buffer:
            yolo_results.extend(
                self.model.predict(buffer, conf=self.conf, verbose=False)
            )

        if not yolo_results:
            return []

        tracks = self.tracks_from_yolo_results(yolo_results)
        tracks = self.remove_wrong_detections(tracks)
        tracks = self.interpolate_ball_positions(tracks)
        save_stub(stub_path, tracks)
        return tracks
