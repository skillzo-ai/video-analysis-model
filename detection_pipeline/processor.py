import cv2
import supervision as sv
import numpy as np
from .detector import Detector
from .visualize import Visualizer
from .reid import PlayerReID, cosine_similarity
from .ball_kalman import BallKalmanTracker
from .possession import PossessionAssigner

class VideoProcessor:
    def __init__(self, player_model_path: str, ball_model_path: str, output_path: str = "output_tracked.mp4"):
        self.detector = Detector(player_model_path, ball_model_path)
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

    def process_video(self, source: str):
        """
        Process the video frame-by-frame with tracking and visual Re-ID.
        """
        video_info = sv.VideoInfo.from_video_path(source)
        frame_generator = sv.get_video_frames_generator(source)
        
        with sv.VideoSink(self.output_path, video_info) as sink:
            for frame in frame_generator:
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

                if np.any(ball_mask):
                    ball_idx = int(np.where(ball_mask)[0][0])
                    b = detections.xyxy[ball_idx].astype(np.float32)
                    bc = ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)
                    self.ball_tracker.last_size_wh = (float(b[2] - b[0]), float(b[3] - b[1]))
                    ball_center_xy = self.ball_tracker.update(bc)

                    # Force stable ball id for downstream logic/visuals
                    try:
                        detections.tracker_id[ball_idx] = 0
                    except Exception:
                        pass
                else:
                    # No detection: predict ball location and inject a synthetic ball detection
                    if not self.ball_tracker.initialized:
                        # Must have a ball every frame: initialize at frame center until first real detection arrives.
                        h, w = frame.shape[:2]
                        ball_center_xy = self.ball_tracker.update((w * 0.5, h * 0.5))
                        bb = self.ball_tracker.center_to_bbox_xyxy(ball_center_xy, default_wh=(16.0, 16.0))
                        synth = sv.Detections(
                            xyxy=np.array([bb], dtype=np.float32),
                            confidence=np.array([0.01], dtype=np.float32),
                            class_id=np.array([0], dtype=np.int32),
                            tracker_id=np.array([0], dtype=np.int32),
                        )
                        detections = self._append_synthetic_detection(detections, synth)
                    else:
                        ball_center_xy = self.ball_tracker.predict(dt=1.0)
                        if ball_center_xy[0] is not None:
                            bb = self.ball_tracker.center_to_bbox_xyxy(ball_center_xy, default_wh=(16.0, 16.0))
                            synth = sv.Detections(
                                xyxy=np.array([bb], dtype=np.float32),
                                confidence=np.array([0.01], dtype=np.float32),
                                class_id=np.array([0], dtype=np.int32),
                                tracker_id=np.array([0], dtype=np.int32),
                            )
                            detections = self._append_synthetic_detection(detections, synth)
                
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
                labels = []
                for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
                    class_name = {0: "Ball", 2: "Hoop", 4: "Player"}.get(class_id, "Unknown")
                    labels.append(f"{class_name} #{tracker_id}")
                
                # 5. Visualization
                annotated_frame = self.visualizer.draw_detections(
                    frame=frame, 
                    detections=detections, 
                    labels=labels
                )

                # Top overlay: who has the ball
                if possessor_id is not None:
                    text = f"Player {int(possessor_id)} has ball"
                else:
                    text = "No possession"

                if possessor_id != self._last_possessor_id and possessor_id is not None:
                    print(f"Player {int(possessor_id)} has ball")
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
