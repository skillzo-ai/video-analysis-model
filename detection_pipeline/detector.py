import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

class Detector:
    def __init__(self, player_model_path: str = "best.pt", ball_model_path: str = "ball_detector_model.pt"):
        """
        Initialize with two models:
        - player_model: Primarily for Players (4) and Hoops (2).
        - ball_model: Dedicated for Balls (typically class 0).
        """
        self.player_model = YOLO(player_model_path)
        self.ball_model = YOLO(ball_model_path)
        
        self.player_hoop_classes = [2, 4]
        # Assuming ball_model's class 0 is the ball. 
        # If it's a dedicated model, we might take all its detections.
        self.ball_class = [0] 

    def get_detections(self, frame: np.ndarray, conf: float = 0.3):
        """
        Combine detections from both models.
        """
        # 1. Get Players and Hoops
        player_results = self.player_model(frame, conf=conf, imgsz=832, verbose=False)[0]
        player_detections = sv.Detections.from_ultralytics(player_results)
        player_mask = np.isin(player_detections.class_id, self.player_hoop_classes)
        player_detections = player_detections[player_mask]

        # 2. Get Ball(s)
        ball_results = self.ball_model(frame, conf=conf, imgsz=832, verbose=False)[0]
        ball_detections = sv.Detections.from_ultralytics(ball_results)
        ball_mask = np.isin(ball_detections.class_id, self.ball_class)
        ball_detections = ball_detections[ball_mask]

        # 3. Merge
        combined_detections = sv.Detections.merge([player_detections, ball_detections])
        return combined_detections
