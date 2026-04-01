import cv2
import supervision as sv
import numpy as np
from .detector import Detector
from .visualize import Visualizer
from .reid import PlayerReID, cosine_similarity

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
                
                # 6. Write frame
                sink.write_frame(annotated_frame)
                
        print(f"Tracking complete. Saved to: {self.output_path}")
