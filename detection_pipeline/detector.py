import numpy as np
from ultralytics import YOLO
import supervision as sv

class Detector:
    def __init__(self, model_path: str = "best.pt"):
        """
        Single-model detector for all classes (Ball=0, Hoop=2, Player=4).
        """
        self.model = YOLO(model_path)
        
        self.keep_classes = [0, 2, 4]
        self.player_hoop_classes = [2, 4]
        self.ball_class = [0]

    def get_detections(
        self,
        frame: np.ndarray,
        conf: float = 0.15,
        iou: float = 0.5,
        player_conf: float = 0.3,
    ):
        """
        Run one model, keep Ball/Hoop/Player, and apply a higher conf threshold to players/hoops.
        """
        results = self.model(frame, conf=conf, iou=iou, imgsz=832, verbose=False, classes=self.keep_classes)[0]
        det = sv.Detections.from_ultralytics(results)

        if len(det) == 0:
            return det

        # Keep only desired classes
        keep_mask = np.isin(det.class_id, self.keep_classes)
        det = det[keep_mask]

        # Apply higher conf threshold to players/hoops (ball keeps lower threshold)
        if det.confidence is not None:
            is_player_or_hoop = np.isin(det.class_id, self.player_hoop_classes)
            hi_ok = (~is_player_or_hoop) | (det.confidence >= float(player_conf))
            det = det[hi_ok]

        return det
