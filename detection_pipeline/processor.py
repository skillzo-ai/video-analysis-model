import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from .ball_tracker import BallTracker
from .detector import Detector, _merge_detections
from .visualize import Visualizer
from .possession import PossessionAssigner
from .yolo_export import detections_to_yolo_txt, write_classes_txt

from team_clustering.config import TeamClusteringConfig
from team_clustering.temporal_team_classification import (
    TemporalTeamClassifier,
    TemporalTeamConfig,
)
from team_clustering.visualization import draw_player_ellipse

from core.tracking import assign_ball_owner
from models.data_structures import BallState, Hoop, PlayerState
from tracking import FrameTrackingPipeline, TrackingConfig


class VideoProcessor:
    def __init__(
        self,
        model_path: str,
        output_path: str = "output_tracked.mp4",
        *,
        ball_model_path: str = "ball_detector_model.pt",
        ball_read_stub: bool = False,
        ball_stub_path: str | None = None,
        pass_detector=None,
        shot_detector=None,
        make_miss_detector=None,
        log_events_all_frames: bool = False,
        tracking_config: TrackingConfig | None = None,
        temporal_team_config: TemporalTeamConfig | None = None,
        export_dataset_dir: str | Path | None = None,
        export_interval: int = 25,
    ):
        self.detector = Detector(model_path)
        self.ball_pipeline = BallTracker(ball_model_path)
        self._ball_read_stub = bool(ball_read_stub)
        self._ball_stub_path = ball_stub_path
        self.visualizer = Visualizer()
        self.output_path = output_path
        self.frame_tracking = FrameTrackingPipeline(tracking_config or TrackingConfig())
        self.possession = PossessionAssigner(
            max_dist_px=140.0, switch_confirm_frames=3, keep_frames_when_lost=10
        )
        self._last_possessor_id = None
        # Per-frame delta of *tracked* ball center for event detectors (stable vs raw meas. pairs).
        self._prev_ball_center_events: tuple[float, float] | None = None

        self.pass_detector = pass_detector
        self.shot_detector = shot_detector
        self.make_miss_detector = make_miss_detector
        self.log_events_all_frames = bool(log_events_all_frames)

        # Two-team classification: HSV KMeans + temporal majority vote (see temporal_team_classification).
        self.team_classifier = TemporalTeamClassifier(
            temporal_team_config or TemporalTeamConfig()
        )
        self._team_stats = {
            "Team A": {"passes": 0, "shots": 0, "makes": 0},
            "Team B": {"passes": 0, "shots": 0, "makes": 0},
        }

        self._export_dataset_dir: Path | None = (
            Path(export_dataset_dir).resolve() if export_dataset_dir else None
        )
        self._export_interval = max(1, int(export_interval))
        self._export_dirs_ready = False

    def _ensure_export_layout(self) -> tuple[Path, Path]:
        """Create ``{export}/images`` and ``{export}/labels`` and write ``classes.txt`` once."""
        assert self._export_dataset_dir is not None
        base = self._export_dataset_dir
        images_dir = base / "images"
        labels_dir = base / "labels"
        if not self._export_dirs_ready:
            base.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            names = {int(k): str(v) for k, v in self.detector.main_model.names.items()}
            write_classes_txt(names, base / "classes.txt")
            self._export_dirs_ready = True
        return images_dir, labels_dir

    def _write_dataset_sample(
        self,
        frame_bgr: np.ndarray,
        detections: sv.Detections,
        frame_idx: int,
    ) -> None:
        """Save raw frame + YOLO labels from current detections (no synthetic ball yet)."""
        if self._export_dataset_dir is None:
            return
        if (frame_idx - 1) % self._export_interval != 0:
            return
        images_dir, labels_dir = self._ensure_export_layout()
        h, w = frame_bgr.shape[:2]
        stem = f"frame_{frame_idx:06d}"
        img_path = images_dir / f"{stem}.jpg"
        lbl_path = labels_dir / f"{stem}.txt"
        cv2.imwrite(str(img_path), frame_bgr)
        lbl_path.write_text(detections_to_yolo_txt(detections, w, h), encoding="utf-8")

    @staticmethod
    def _append_synthetic_detection(
        base: sv.Detections, extra: sv.Detections
    ) -> sv.Detections:
        """
        supervision.Detections.merge() requires matching data keys; this append pads base.data keys.
        Assumes `extra` contains exactly one row.
        """
        if base is None or len(base) == 0:
            return extra

        # Core fields
        xyxy = np.concatenate([base.xyxy, extra.xyxy], axis=0)

        confidence = None
        if base.confidence is not None or extra.confidence is not None:
            b = (
                base.confidence
                if base.confidence is not None
                else np.zeros((len(base),), dtype=np.float32)
            )
            e = (
                extra.confidence
                if extra.confidence is not None
                else np.zeros((len(extra),), dtype=b.dtype)
            )
            confidence = np.concatenate([b, e], axis=0)

        class_id = None
        if base.class_id is not None or extra.class_id is not None:
            b = (
                base.class_id
                if base.class_id is not None
                else np.zeros((len(base),), dtype=np.int32)
            )
            e = (
                extra.class_id
                if extra.class_id is not None
                else np.zeros((len(extra),), dtype=b.dtype)
            )
            class_id = np.concatenate([b, e], axis=0)

        tracker_id = None
        if base.tracker_id is not None or extra.tracker_id is not None:
            b = (
                base.tracker_id
                if base.tracker_id is not None
                else np.full((len(base),), -1, dtype=np.int32)
            )
            e = (
                extra.tracker_id
                if extra.tracker_id is not None
                else np.full((len(extra),), -1, dtype=b.dtype)
            )
            tracker_id = np.concatenate([b, e], axis=0)

        # Optional masks
        mask = None
        if (
            getattr(base, "mask", None) is not None
            or getattr(extra, "mask", None) is not None
        ):
            b = base.mask if getattr(base, "mask", None) is not None else None
            e = extra.mask if getattr(extra, "mask", None) is not None else None
            if b is not None and e is not None:
                mask = np.concatenate([b, e], axis=0)
            elif b is not None:
                # pad one empty mask if needed (best-effort)
                mask = b
            else:
                mask = e

        # Pad data keys
        base_data = dict(getattr(base, "data", {}) or {})
        extra_data = dict(getattr(extra, "data", {}) or {})
        out_data = {}
        all_keys = set(base_data.keys()) | set(extra_data.keys())
        for k in all_keys:
            bv = base_data.get(k, None)
            ev = extra_data.get(k, None)

            if isinstance(bv, np.ndarray):
                if ev is None:
                    pad = np.zeros((1, *bv.shape[1:]), dtype=bv.dtype)
                    out_data[k] = np.concatenate([bv, pad], axis=0)
                else:
                    out_data[k] = np.concatenate([bv, np.asarray(ev)], axis=0)
            elif isinstance(bv, (list, tuple)):
                if ev is None:
                    out_data[k] = list(bv) + [None]
                elif isinstance(ev, (list, tuple)):
                    out_data[k] = list(bv) + list(ev)
                else:
                    out_data[k] = list(bv) + [ev]
            else:
                # Unknown / missing in base: create placeholder list for base rows
                base_pad = [None] * len(base)
                if isinstance(ev, (list, tuple)):
                    out_data[k] = base_pad + list(ev)
                else:
                    out_data[k] = base_pad + [ev]

        return sv.Detections(
            xyxy=xyxy,
            mask=mask,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
            data=out_data,
        )

    def _team_label_for_player(self, player_id: int) -> str:
        return self.team_classifier.get_team(int(player_id))

    def export_team_track_json(self) -> dict[str, str]:
        """``track_id`` -> ``Team A`` / ``Team B`` (for JSON export alongside video)."""
        return self.team_classifier.export_track_team_json()

    def _bump_team_stat(self, team: str, stat_key: str) -> None:
        bucket = self._team_stats.get(team)
        if bucket is not None and stat_key in bucket:
            bucket[stat_key] += 1

    def team_stats_export_dict(self) -> dict:
        """Stats for JSON export (keys team_A / team_B)."""
        a = self._team_stats["Team A"]
        b = self._team_stats["Team B"]
        return {
            "team_A": {
                "passes": int(a["passes"]),
                "shots": int(a["shots"]),
                "makes": int(a["makes"]),
            },
            "team_B": {
                "passes": int(b["passes"]),
                "shots": int(b["shots"]),
                "makes": int(b["makes"]),
            },
        }

    @staticmethod
    def _draw_team_stats_panel(
        bgr: np.ndarray,
        stats_a: dict[str, int],
        stats_b: dict[str, int],
        color_a: tuple[int, int, int],
        color_b: tuple[int, int, int],
    ) -> None:
        h, w = bgr.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        thick = 1
        line_h = 21
        margin = 10
        pad = 6

        def lines_for(
            team_name: str, st: dict[str, int]
        ) -> list[tuple[str, tuple[int, int, int]]]:
            hdr_color = color_a if team_name == "Team A" else color_b
            return [
                (team_name, hdr_color),
                (f"Passes: {st['passes']}", (240, 240, 240)),
                (f"Shots:  {st['shots']}", (240, 240, 240)),
                (f"Makes:  {st['makes']}", (240, 240, 240)),
            ]

        block_a = lines_for("Team A", stats_a)
        block_b = lines_for("Team B", stats_b)
        gap_lines = 1
        all_rows: list[tuple[str, tuple[int, int, int]]] = (
            block_a + [("", (0, 0, 0))] * gap_lines + block_b
        )

        max_tw = 0
        for text, _ in all_rows:
            if not text:
                continue
            (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
            max_tw = max(max_tw, tw)

        total_h = sum(line_h if t else line_h // 2 for t, _ in all_rows) + pad * 2
        x1 = int(w - margin - max_tw - pad * 2)
        y1 = margin
        x2 = w - margin
        y2 = min(h - margin, y1 + total_h)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (28, 28, 32), -1)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (72, 72, 78), 1)

        cx = x1 + pad
        cy = y1 + pad + 16
        for text, color in all_rows:
            if not text:
                cy += line_h // 2
                continue
            (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
            tx = x2 - pad - tw
            cv2.putText(bgr, text, (tx, cy), font, scale, color, thick, cv2.LINE_AA)
            cy += line_h

    @staticmethod
    def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float32).reshape(4)
        b = b.astype(np.float32).reshape(4)
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        iw = max(0.0, x2 - x1)
        ih = max(0.0, y2 - y1)
        inter = iw * ih
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    def process_video(self, source: str):
        """
        Process the video frame-by-frame with tracking and visual Re-ID.

        Ball: batched ``ball_detector_model.pt`` pass (filter + interpolate), then merged per frame.
        """
        video_info = sv.VideoInfo.from_video_path(source)
        video_fps = float(getattr(video_info, "fps", None) or 0.0)
        if video_fps <= 1e-6:
            video_fps = 30.0

        ball_tracks = self.ball_pipeline.run_ball_pipeline_on_video(
            source,
            read_from_stub=self._ball_read_stub,
            stub_path=self._ball_stub_path,
        )

        frame_generator = sv.get_video_frames_generator(source)

        def batched(iterable, n):
            batch = []
            for item in iterable:
                batch.append(item)
                if len(batch) == n:
                    yield batch
                    batch = []
            if batch:
                yield batch

        with sv.VideoSink(self.output_path, video_info, codec="mp4v") as sink:
            frame_idx = 0
            for frames_batch in batched(frame_generator, 8):
                detections_batch = self.detector.get_detections_batch(frames_batch)
                for frame, detections in zip(frames_batch, detections_batch):
                    frame_idx += 1
                    time_sec = (frame_idx - 1) / video_fps

                    # 1b. Merge ball from BallTracker pipeline (highest-conf ``Ball`` + temporal cleanup)
                    ti = frame_idx - 1
                    if ti < len(ball_tracks):
                        bb = ball_tracks[ti].get(1, {}).get("bbox", [])
                        if isinstance(bb, (list, tuple)) and len(bb) == 4:
                            ball_det = sv.Detections(
                                xyxy=np.asarray([bb], dtype=np.float32),
                                confidence=np.array([0.99], dtype=np.float32),
                                class_id=np.array([0], dtype=np.int32),
                            )
                            detections = _merge_detections(detections, ball_det)

                    # 2. ByteTrack + Re-ID (appearance / jersey / height) + optional Kalman (reusable module)
                    detections, _frame_track_result = self.frame_tracking.step(
                        frame, detections, frame_index=frame_idx
                    )

                    # --- NEW: Ball Filtering Logic ---
                    # If there are multiple balls, keep only the one nearest to the highest-confidence player
                    ball_mask = detections.class_id == 0
                    player_mask = detections.class_id == 4

                    if np.sum(ball_mask) > 1:
                        if np.sum(player_mask) > 0:
                            # Find player with highest confidence
                            player_idx = np.argmax(detections.confidence[player_mask])
                            player_bbox = detections.xyxy[player_mask][player_idx]
                            player_center = np.array(
                                [
                                    (player_bbox[0] + player_bbox[2]) / 2,
                                    (player_bbox[1] + player_bbox[3]) / 2,
                                ]
                            )

                            # Find nearest ball
                            ball_indices = np.where(ball_mask)[0]
                            min_dist = float("inf")
                            best_ball_idx = ball_indices[0]

                            for b_idx in ball_indices:
                                ball_bbox = detections.xyxy[b_idx]
                                ball_center = np.array(
                                    [
                                        (ball_bbox[0] + ball_bbox[2]) / 2,
                                        (ball_bbox[1] + ball_bbox[3]) / 2,
                                    ]
                                )
                                dist = np.linalg.norm(player_center - ball_center)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_ball_idx = b_idx

                            # Filter out other balls
                            final_mask = np.ones(len(detections), dtype=bool)
                            for b_idx in ball_indices:
                                if b_idx != best_ball_idx:
                                    final_mask[b_idx] = False
                            detections = detections[final_mask]
                        else:
                            # No players? Keep only highest confidence ball
                            ball_indices = np.where(ball_mask)[0]
                            best_ball_idx = ball_indices[
                                np.argmax(detections.confidence[ball_mask])
                            ]
                            final_mask = np.ones(len(detections), dtype=bool)
                            for b_idx in ball_indices:
                                if b_idx != best_ball_idx:
                                    final_mask[b_idx] = False
                            detections = detections[final_mask]
                    # ---------------------------------

                    # 2b. Dataset export (same pass — before synthetic ball placeholder)
                    self._write_dataset_sample(frame, detections, frame_idx)

                    # 3. Ball center from merged ball row (BallTracker + interpolate); synthetic if missing
                    ball_mask = detections.class_id == 0
                    ball_center_xy = None
                    ball_source = "none"

                    if np.any(ball_mask):
                        ball_idx = int(np.where(ball_mask)[0][0])
                        b = detections.xyxy[ball_idx].astype(np.float32)
                        ball_center_xy = (
                            float((b[0] + b[2]) * 0.5),
                            float((b[1] + b[3]) * 0.5),
                        )
                        ball_source = "ball_pipeline"
                        try:
                            detections.tracker_id[ball_idx] = 0
                        except Exception:
                            pass
                    else:
                        h, w = frame.shape[:2]
                        ball_center_xy = (float(w * 0.5), float(h * 0.5))
                        ball_source = "init_center"
                        bb = np.array(
                            [
                                float(w * 0.5 - 8),
                                float(h * 0.5 - 8),
                                float(w * 0.5 + 8),
                                float(h * 0.5 + 8),
                            ],
                            dtype=np.float32,
                        )
                        synth = sv.Detections(
                            xyxy=np.array([bb], dtype=np.float32),
                            confidence=np.array([0.01], dtype=np.float32),
                            class_id=np.array([0], dtype=np.int32),
                            tracker_id=np.array([0], dtype=np.int32),
                        )
                        detections = self._append_synthetic_detection(detections, synth)

                    if ball_center_xy is not None and ball_center_xy[0] is not None:
                        bx, by = float(ball_center_xy[0]), float(ball_center_xy[1])

                    # 3.6 Possession assignment (nearest player to ball)
                    player_mask = detections.class_id == 4
                    possessor_id = None
                    if ball_center_xy is not None and np.any(player_mask):
                        possessor_id = self.possession.update(
                            ball_center_xy=ball_center_xy,
                            player_xyxy=detections.xyxy[player_mask],
                            player_ids=detections.tracker_id[player_mask],
                        )
                    else:
                        possessor_id = self.possession.update(
                            ball_center_xy=None,
                            player_xyxy=None,
                            player_ids=None,
                        )

                    # 3.7 Basketball event detectors (optional plug-in)
                    if (
                        self.pass_detector is not None
                        and self.shot_detector is not None
                        and self.make_miss_detector is not None
                        and ball_center_xy is not None
                        and ball_center_xy[0] is not None
                    ):
                        bx, by = float(ball_center_xy[0]), float(ball_center_xy[1])
                        if self._prev_ball_center_events is not None:
                            px, py = self._prev_ball_center_events
                            bvx, bvy = bx - px, by - py
                        else:
                            bvx, bvy = 0.0, 0.0
                        self._prev_ball_center_events = (bx, by)
                        ball_state = BallState(position=(bx, by), velocity=(bvx, bvy))

                        player_states: list[PlayerState] = []
                        if np.any(player_mask):
                            for j in np.where(player_mask)[0]:
                                bb = detections.xyxy[int(j)].astype(np.float32)
                                cx = (float(bb[0]) + float(bb[2])) * 0.5
                                cy = (float(bb[1]) + float(bb[3])) * 0.5
                                pid = int(detections.tracker_id[int(j)])
                                player_states.append(
                                    PlayerState(player_id=pid, position=(cx, cy))
                                )

                        centers = [p.position for p in player_states]
                        pids = [p.player_id for p in player_states]
                        owner_for_events, _ = (
                            assign_ball_owner((bx, by), centers, pids)
                            if player_states
                            else (None, None)
                        )

                        hoop_state = None
                        hoop_mask = detections.class_id == 2
                        if np.any(hoop_mask):
                            hi = int(np.where(hoop_mask)[0][0])
                            hbb = detections.xyxy[hi].astype(np.float32)
                            hoop_state = Hoop(
                                bbox=(
                                    float(hbb[0]),
                                    float(hbb[1]),
                                    float(hbb[2]),
                                    float(hbb[3]),
                                )
                            )

                        pass_evt = self.pass_detector.detect(
                            ball_state, player_states, owner_for_events
                        )
                        shot_evt = self.shot_detector.detect(ball_state, hoop_state)
                        make_evt = self.make_miss_detector.detect(ball_state, hoop_state)

                        payload = {
                            "frame": frame_idx,
                            "pass": bool(pass_evt),
                            "shot": bool(shot_evt),
                            "make": bool(make_evt),
                        }

                        if pass_evt and self.pass_detector.last_pass_from_id is not None:
                            tid_pass = int(self.pass_detector.last_pass_from_id)
                            self._bump_team_stat(
                                self._team_label_for_player(tid_pass), "passes"
                            )
                        if shot_evt and owner_for_events is not None:
                            self._bump_team_stat(
                                self._team_label_for_player(int(owner_for_events)), "shots"
                            )
                        if make_evt and owner_for_events is not None:
                            self._bump_team_stat(
                                self._team_label_for_player(int(owner_for_events)), "makes"
                            )

                    if ball_center_xy is None or ball_source == "init_center":
                        self._prev_ball_center_events = None

                    # 4. Label preparation
                    # Keep the video overlay clean: no per-track text labels in the final render.
                    labels = None

                    # 5. Visualization (ball/hoop via supervision; players drawn with team colors)
                    annotated_frame = self.visualizer.draw_detections(
                        frame=frame, detections=detections, labels=labels
                    )

                    # 5.1 Robust two-team classification (HSV KMeans + temporal majority) + foot ellipses (legacy style)
                    player_mask = detections.class_id == 4
                    possessor_id_int = (
                        int(possessor_id) if possessor_id is not None else None
                    )
                    tcfg = self.team_classifier.cfg
                    ellipse_cfg = TeamClusteringConfig(
                        debug=False,
                        draw_text=False,
                        ellipse_color_team_a_bgr=tcfg.color_team_a_bgr,
                        ellipse_color_team_b_bgr=tcfg.color_team_b_bgr,
                        possession_highlight_bgr=tcfg.possession_highlight_bgr,
                    )
                    if np.any(player_mask):
                        player_xyxy = detections.xyxy[player_mask].astype(np.float32)
                        player_ids = detections.tracker_id[player_mask].astype(int)
                        confs = (
                            detections.confidence[player_mask]
                            if detections.confidence is not None
                            else None
                        )
                        team_labels, _ = self.team_classifier.update_frame(
                            annotated_frame,
                            player_xyxy,
                            player_ids,
                            confidences=confs,
                            time_sec=time_sec,
                        )
                        bboxes_xywh: list[list[float]] = []
                        for bb in player_xyxy:
                            x1, y1, x2, y2 = bb.tolist()
                            bboxes_xywh.append(
                                [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]
                            )

                        for bb_xywh, team_smoothed in zip(bboxes_xywh, team_labels):
                            annotated_frame = draw_player_ellipse(
                                annotated_frame,
                                bb_xywh,
                                team_smoothed,
                                ellipse_cfg,
                                player_id=None,
                            )

                    self._last_possessor_id = possessor_id

                    self._draw_team_stats_panel(
                        annotated_frame,
                        self._team_stats["Team A"],
                        self._team_stats["Team B"],
                        tcfg.color_team_a_bgr,
                        tcfg.color_team_b_bgr,
                    )

                    # 6. Write frame
                    sink.write_frame(annotated_frame)

        # Majority lock for short clips + final labels for JSON export
        self.team_classifier.finalize_teams()
