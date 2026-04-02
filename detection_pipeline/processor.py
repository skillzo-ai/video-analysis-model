import cv2
import supervision as sv
import numpy as np
from .detector import Detector
from .visualize import Visualizer
from .reid import PlayerReID, cosine_similarity
from .ball_kalman import BallKalmanTracker
from .possession import PossessionAssigner

from team_clustering.config import TeamClusteringConfig
from team_clustering.pipeline import classify_teams_no_draw
from team_clustering.visualization import draw_player_ellipse

class VideoProcessor:
    def __init__(self, model_path: str, output_path: str = "output_tracked.mp4"):
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
        self.ball_tracker = BallKalmanTracker(process_var=25.0, meas_var=80.0)
        self.possession = PossessionAssigner(max_dist_px=140.0, switch_confirm_frames=3, keep_frames_when_lost=10)
        self._last_possessor_id = None

        # Team clustering state (tracker_id -> vote counts)
        self.team_cfg = TeamClusteringConfig(debug=False, draw_text=False)
        self._team_votes = {}  # tid -> np.array([votes_teamA, votes_teamB], float32)

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
        
        with sv.VideoSink(self.output_path, video_info) as sink:
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

                # 3.5 Ball Kalman tracking (ensure ball exists every frame)
                ball_mask = detections.class_id == 0
                ball_center_xy = None
                ball_source = "none"

                if np.any(ball_mask):
                    ball_idx = int(np.where(ball_mask)[0][0])
                    b = detections.xyxy[ball_idx].astype(np.float32)
                    bc = ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)
                    self.ball_tracker.last_size_wh = (float(b[2] - b[0]), float(b[3] - b[1]))

                    # Accept detection; tracker's filtered center is what we draw/use.
                    ball_center_xy = self.ball_tracker.update(bc)
                    self.ball_tracker.note_measurement(bc)
                    ball_source = "det"

                    # Force displayed bbox to follow tracker output (prevents "stuck box")
                    tracked_bb = self.ball_tracker.center_to_bbox_xyxy(ball_center_xy, default_wh=(16.0, 16.0))
                    detections.xyxy[ball_idx] = tracked_bb

                    # Force stable ball id for downstream logic/visuals
                    try:
                        detections.tracker_id[ball_idx] = 0
                    except Exception:
                        pass
                else:
                    # No detection: predicted = current_pos + (current_pos - prev_pos)
                    if not self.ball_tracker.initialized:
                        # Must have a ball every frame: initialize at frame center until first real detection arrives.
                        h, w = frame.shape[:2]
                        ball_center_xy = self.ball_tracker.update((w * 0.5, h * 0.5))
                        ball_source = "init_center"
                        bb = self.ball_tracker.center_to_bbox_xyxy(ball_center_xy, default_wh=(16.0, 16.0))
                        synth = sv.Detections(
                            xyxy=np.array([bb], dtype=np.float32),
                            confidence=np.array([0.01], dtype=np.float32),
                            class_id=np.array([0], dtype=np.int32),
                            tracker_id=np.array([0], dtype=np.int32),
                        )
                        detections = self._append_synthetic_detection(detections, synth)
                    else:
                        vel_pred_center = self.ball_tracker.velocity_predict_center()
                        if vel_pred_center is None:
                            # fallback to Kalman predict if we don't have two centers yet
                            ball_center_xy = self.ball_tracker.predict(dt=1.0)
                            ball_source = "kalman_pred"
                        else:
                            ball_center_xy = self.ball_tracker.update(vel_pred_center)
                            ball_source = "vel_pred"

                        if ball_center_xy is not None and ball_center_xy[0] is not None:
                            bb = self.ball_tracker.center_to_bbox_xyxy(ball_center_xy, default_wh=(16.0, 16.0))
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
                
                # 6. Write frame
                sink.write_frame(annotated_frame)
                
        print(f"Tracking complete. Saved to: {self.output_path}")
