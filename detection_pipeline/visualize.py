import cv2
import numpy as np
import supervision as sv

class Visualizer:
    def __init__(self):
        """
        Visualizer for non-player objects (ball/hoop).

        Player visualization is handled by `team_clustering` to ensure
        team-specific ellipses and label coloring.
        """
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
        Annotate frame with boxes for ball/hoop.
        Players (class 4) are intentionally not drawn here.
        """
        annotated_frame = frame.copy()
        
        # Keep only non-player detections (ball/hoop/etc).
        other_mask = detections.class_id != 4
        other_detections = detections[other_mask]

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
