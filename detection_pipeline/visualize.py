import cv2
import numpy as np
import supervision as sv

class Visualizer:
    def __init__(self):
        """
        Custom visualizer with ellipses for players.
        """
        self.ellipse_annotator = sv.EllipseAnnotator(
            thickness=2
        )
        self.box_annotator = sv.BoxAnnotator(
            thickness=2
        )
        self.label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.TOP_CENTER,
            text_thickness=1,
            text_scale=0.5
        )

    def draw_detections(self, frame: np.ndarray, detections: sv.Detections, labels: list = None):
        """
        Annotate frame with ellipses for players and boxes for ball/hoop.
        """
        annotated_frame = frame.copy()
        
        # Split detections: Players (class 4) vs others
        player_mask = detections.class_id == 4
        other_mask = ~player_mask
        
        player_detections = detections[player_mask]
        other_detections = detections[other_mask]
        
        # Draw ellipses for players
        if len(player_detections) > 0:
            annotated_frame = self.ellipse_annotator.annotate(
                scene=annotated_frame, 
                detections=player_detections
            )
        
        # Draw boxes for ball/hoop
        if len(other_detections) > 0:
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame, 
                detections=other_detections
            )
            
        # Draw labels
        if labels:
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame, 
                detections=detections,
                labels=labels
            )
            
        return annotated_frame
