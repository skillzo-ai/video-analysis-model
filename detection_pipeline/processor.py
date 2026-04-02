import json
import cv2
import supervision as sv
import numpy as np
from .detector import Detector
from .visualize import Visualizer
from .reid import PlayerReID, cosine_similarity
from .ball_deepsort import BallDeepOcSortTracker
from .possession import PossessionAssigner

from team_clustering.config import TeamClusteringConfig
from team_clustering.pipeline import classify_teams_no_draw
from team_clustering.visualization import draw_player_ellipse

from core.tracking import assign_ball_owner
from models.data_structures import BallState, Hoop, PlayerState


class VideoProcessor:
    def __init__(
        self,
        model_path: str,
        output_path: str = "output_tracked.mp4",
        *,
        pass_detector=None,
        shot_detector=None,
        make_miss_detector=None,
        log_events_all_frames: bool = False,
    ):
        self.detector = Detector(model_path)
        self.visualizer = Visualizer()
        self.output_path = output_path
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25, 
            lost_track_buffer=150, # Increased further
            minimum_matching_threshold=0.8
        )
        self.reid = PlayerReID()
        
        # ID memory
        self.id_memory = {} # tracker_id -> list of embeddings
        self.id_map = {}    # current_tracker_id -> original_id (for Re-ID remapping)
        self.next_reid_id = 1
        self.similarity_threshold = 0.85 # Threshold for visual match
        # DeepOcSort: Kalman bbox + ReID association (BoxMOT; closest to DeepSORT in this stack).
        self.ball_tracker = BallDeepOcSortTracker()
        self.possession = PossessionAssigner(max_dist_px=140.0, switch_confirm_frames=3, keep_frames_when_lost=10)
        self._last_possessor_id = None
        # Per-frame delta of *tracked* ball center for event detectors (stable vs raw meas. pairs).
        self._prev_ball_center_events: tuple[float, float] | None = None

        self.pass_detector = pass_detector
        self.shot_detector = shot_detector
        self.make_miss_detector = make_miss_detector
        self.log_events_all_frames = bool(log_events_all_frames)

        # Team clustering state (tracker_id -> vote counts)
        self.team_cfg = TeamClusteringConfig(debug=False, draw_text=False)
        self._team_votes = {}  # tid -> np.array([votes_teamA, votes_teamB], float32)
        self._team_stats = {
            "Team A": {"passes": 0, "shots": 0, "makes": 0},
            "Team B": {"passes": 0, "shots": 0, "makes": 0},
        }

    @staticmethod
    def _append_synthetic_detection(base: sv.Detections, extra: sv.Detections) -> sv.Detections:
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
            b = base.confidence if base.confidence is not None else np.zeros((len(base),), dtype=np.float32)
            e = extra.confidence if extra.confidence is not None else np.zeros((len(extra),), dtype=b.dtype)
            confidence = np.concatenate([b, e], axis=0)

        class_id = None
        if base.class_id is not None or extra.class_id is not None:
            b = base.class_id if base.class_id is not None else np.zeros((len(base),), dtype=np.int32)
            e = extra.class_id if extra.class_id is not None else np.zeros((len(extra),), dtype=b.dtype)
            class_id = np.concatenate([b, e], axis=0)

        tracker_id = None
        if base.tracker_id is not None or extra.tracker_id is not None:
            b = base.tracker_id if base.tracker_id is not None else np.full((len(base),), -1, dtype=np.int32)
            e = extra.tracker_id if extra.tracker_id is not None else np.full((len(extra),), -1, dtype=b.dtype)
            tracker_id = np.concatenate([b, e], axis=0)

        # Optional masks
        mask = None
        if getattr(base, "mask", None) is not None or getattr(extra, "mask", None) is not None:
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
        votes = self._team_votes.get(int(player_id))
        if votes is None:
            return "Team A"
        return "Team A" if float(votes[0]) >= float(votes[1]) else "Team B"

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

        def lines_for(team_name: str, st: dict[str, int]) -> list[tuple[str, tuple[int, int, int]]]:
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
        all_rows: list[tuple[str, tuple[int, int, int]]] = block_a + [("", (0, 0, 0))] * gap_lines + block_b

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
        """
        video_info = sv.VideoInfo.from_video_path(source)
        frame_generator = sv.get_video_frames_generator(source)
        
        with sv.VideoSink(self.output_path, video_info, codec="mp4v") as sink:
            frame_idx = 0
            for frame in frame_generator:
                frame_idx += 1
                # 1. Detection
                detections = self.detector.get_detections(frame)
                
                # 2. Tracking
                detections = self.tracker.update_with_detections(detections)
                
                # --- NEW: Ball Filtering Logic ---
                # If there are multiple balls, keep only the one nearest to the highest-confidence player
                ball_mask = detections.class_id == 0
                player_mask = detections.class_id == 4
                
                if np.sum(ball_mask) > 1:
                    if np.sum(player_mask) > 0:
                        # Find player with highest confidence
                        player_idx = np.argmax(detections.confidence[player_mask])
                        player_bbox = detections.xyxy[player_mask][player_idx]
                        player_center = np.array([(player_bbox[0] + player_bbox[2]) / 2, (player_bbox[1] + player_bbox[3]) / 2])
                        
                        # Find nearest ball
                        ball_indices = np.where(ball_mask)[0]
                        min_dist = float('inf')
                        best_ball_idx = ball_indices[0]
                        
                        for b_idx in ball_indices:
                            ball_bbox = detections.xyxy[b_idx]
                            ball_center = np.array([(ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2])
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
                        best_ball_idx = ball_indices[np.argmax(detections.confidence[ball_mask])]
                        final_mask = np.ones(len(detections), dtype=bool)
                        for b_idx in ball_indices:
                            if b_idx != best_ball_idx:
                                final_mask[b_idx] = False
                        detections = detections[final_mask]
                # ---------------------------------

                # 3. Visual Re-ID for players (class 4)
                new_tracker_ids = []
                for i in range(len(detections)):
                    bbox = detections.xyxy[i].astype(int)
                    tid = detections.tracker_id[i]
                    cls = detections.class_id[i]
                    
                    if cls == 4: # Player
                        # Crop and extract embedding
                        crop = frame[max(0, bbox[1]):bbox[3], max(0, bbox[0]):bbox[2]]
                        embedding = self.reid.extract_embedding(crop)
                        
                        if embedding is not None:
                            # Re-ID logic
                            if tid not in self.id_map:
                                # New track ID from ByteTrack. Check if it matches a known player's appearance.
                                matched_id = None
                                best_sim = -1
                                
                                for old_id, saved_embeddings in self.id_memory.items():
                                    # Compare against last 5 embeddings of this player
                                    sims = [cosine_similarity(embedding, e) for e in saved_embeddings[-5:]]
                                    max_sim = max(sims) if sims else 0
                                    
                                    if max_sim > self.similarity_threshold and max_sim > best_sim:
                                        best_sim = max_sim
                                        matched_id = old_id
                                
                                if matched_id is not None:
                                    # Match found! Map this new tracker ID to our stored player ID
                                    self.id_map[tid] = matched_id
                                else:
                                    # Truly new player? Or first time seeing this tracker ID
                                    self.id_map[tid] = tid
                            
                            # Store embedding for the mapped ID
                            mapped_id = self.id_map[tid]
                            if mapped_id not in self.id_memory:
                                self.id_memory[mapped_id] = []
                            self.id_memory[mapped_id].append(embedding)
                            
                        # Use mapped ID
                        new_tracker_ids.append(self.id_map.get(tid, tid))
                    else:
                        # Ball or Hoop - just use original tracker ID
                        new_tracker_ids.append(tid)

                # Update detections with re-mapped IDs
                detections.tracker_id = np.array(new_tracker_ids)

                # 3.5 Ball track (DeepOcSort) + synthetic ball when needed
                ball_mask = detections.class_id == 0
                ball_center_xy = None
                ball_source = "none"

                if np.any(ball_mask):
                    ball_idx = int(np.where(ball_mask)[0][0])
                    b = detections.xyxy[ball_idx].astype(np.float32)
                    conf = (
                        float(detections.confidence[ball_idx])
                        if detections.confidence is not None
                        else 0.25
                    )

                    ball_center_xy, tracked_bb = self.ball_tracker.update(frame, b, conf)
                    if ball_center_xy is None or tracked_bb is None:
                        bc = ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)
                        ball_center_xy = bc
                        tracked_bb = b.copy()
                        ball_source = "det_raw"
                    else:
                        detections.xyxy[ball_idx] = tracked_bb
                        ball_source = "det"

                    try:
                        detections.tracker_id[ball_idx] = 0
                    except Exception:
                        pass
                else:
                    if not self.ball_tracker.initialized:
                        h, w = frame.shape[:2]
                        ball_center_xy = (float(w * 0.5), float(h * 0.5))
                        ball_source = "init_center"
                        bb = self.ball_tracker.center_to_bbox_xyxy(
                            ball_center_xy, default_wh=(16.0, 16.0)
                        )
                        synth = sv.Detections(
                            xyxy=np.array([bb], dtype=np.float32),
                            confidence=np.array([0.01], dtype=np.float32),
                            class_id=np.array([0], dtype=np.int32),
                            tracker_id=np.array([0], dtype=np.int32),
                        )
                        detections = self._append_synthetic_detection(detections, synth)
                    else:
                        ball_center_xy, tracked_bb = self.ball_tracker.update(frame, None, 0.0)
                        ball_source = "deepsort_kalman"

                        if ball_center_xy is not None and tracked_bb is not None:
                            bb = np.asarray(tracked_bb, dtype=np.float32)
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
                            player_states.append(PlayerState(player_id=pid, position=(cx, cy)))

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
                    if self.log_events_all_frames or pass_evt or shot_evt or make_evt:
                        print(json.dumps(payload))

                    if pass_evt and self.pass_detector.last_pass_from_id is not None:
                        tid_pass = int(self.pass_detector.last_pass_from_id)
                        self._bump_team_stat(self._team_label_for_player(tid_pass), "passes")
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
                # We'll draw colored labels ourselves after team assignment.
                labels = None
                
                # 5. Visualization
                annotated_frame = self.visualizer.draw_detections(
                    frame=frame, 
                    detections=detections, 
                    labels=labels
                )

                # 5.1 Team clustering + team-specific ellipses for players
                player_mask = detections.class_id == 4
                possessor_id_int = int(possessor_id) if possessor_id is not None else None
                if np.any(player_mask):
                    player_xyxy = detections.xyxy[player_mask].astype(np.float32)
                    player_ids = detections.tracker_id[player_mask].astype(int)

                    # Convert xyxy -> xywh expected by team_clustering
                    bboxes_xywh = []
                    for bb in player_xyxy:
                        x1, y1, x2, y2 = bb.tolist()
                        bboxes_xywh.append([x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)])

                    tc = classify_teams_no_draw(annotated_frame, bboxes_xywh, self.team_cfg)
                    teams = tc["teams"]

                    # Temporal smoothing via per-track voting
                    for tid, team in zip(player_ids.tolist(), teams):
                        if tid not in self._team_votes:
                            self._team_votes[tid] = np.zeros((2,), dtype=np.float32)

                        if self.team_cfg.temporal_vote_decay and self.team_cfg.temporal_vote_decay > 0:
                            self._team_votes[tid] *= float(1.0 - self.team_cfg.temporal_vote_decay)

                        if team == "Team A":
                            self._team_votes[tid][0] += 1.0
                        else:
                            self._team_votes[tid][1] += 1.0

                    # Draw ellipses with smoothed label per player
                    for bb_xywh, tid in zip(bboxes_xywh, player_ids.tolist()):
                        votes = self._team_votes.get(int(tid), None)
                        if votes is None:
                            team_smoothed = "Team A"
                        else:
                            team_smoothed = "Team A" if float(votes[0]) >= float(votes[1]) else "Team B"
                        annotated_frame = draw_player_ellipse(
                            annotated_frame,
                            bb_xywh,
                            team_smoothed,
                            self.team_cfg,
                            player_id=None,
                        )

                    # Draw player IDs above bbox, colored by team / possession
                    for bb_xyxy, tid in zip(player_xyxy.astype(int).tolist(), player_ids.tolist()):
                        x1, y1, x2, y2 = (int(bb_xyxy[0]), int(bb_xyxy[1]), int(bb_xyxy[2]), int(bb_xyxy[3]))
                        votes = self._team_votes.get(int(tid), None)
                        team_smoothed = "Team A" if votes is None or float(votes[0]) >= float(votes[1]) else "Team B"
                        if possessor_id_int is not None and int(tid) == possessor_id_int:
                            color = self.team_cfg.possession_highlight_bgr
                        else:
                            color = (
                                self.team_cfg.ellipse_color_team_a_bgr
                                if team_smoothed == "Team A"
                                else self.team_cfg.ellipse_color_team_b_bgr
                            )

                        text = f"Player #{int(tid)}"
                        org = (x1, max(0, y1 - 6))
                        cv2.putText(
                            annotated_frame,
                            text,
                            org,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            float(self.team_cfg.text_scale),
                            color,
                            int(self.team_cfg.text_thickness),
                            cv2.LINE_AA,
                        )

                # Draw labels for non-player detections (ball/hoop) in a neutral color
                neutral = (255, 255, 255)
                for bb, cls, tid in zip(detections.xyxy.astype(int), detections.class_id, detections.tracker_id):
                    if int(cls) == 4:
                        continue
                    x1, y1, x2, y2 = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
                    class_name = {0: "Ball", 2: "Hoop"}.get(int(cls), "Obj")
                    text = f"{class_name} #{int(tid)}"
                    cv2.putText(
                        annotated_frame,
                        text,
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        neutral,
                        1,
                        cv2.LINE_AA,
                    )

                # Top overlay: who has the ball
                if possessor_id is not None:
                    text = f"Player {int(possessor_id)} has ball"
                else:
                    text = "No possession"

                
               
               
                self._last_possessor_id = possessor_id

                cv2.putText(
                    annotated_frame,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                self._draw_team_stats_panel(
                    annotated_frame,
                    self._team_stats["Team A"],
                    self._team_stats["Team B"],
                    self.team_cfg.ellipse_color_team_a_bgr,
                    self.team_cfg.ellipse_color_team_b_bgr,
                )

                # 6. Write frame
                sink.write_frame(annotated_frame)
