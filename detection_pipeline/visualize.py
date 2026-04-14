import cv2
import numpy as np
import supervision as sv

# BGR — ball must pop on wood / crowd; default BoxAnnotator palette is easy to lose on broadcast.
_BALL_BGR = (0, 255, 255)  # cyan


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

    @staticmethod
    def _draw_ball_bgr(
        frame_bgr: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        *,
        box_thickness: int = 3,
    ) -> None:
        """Thick high-contrast box + circle so small balls stay visible on video."""
        h, w = frame_bgr.shape[:2]
        x1 = int(np.clip(x1, 0, w - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        y2 = int(np.clip(y2, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            return
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), _BALL_BGR, box_thickness, cv2.LINE_AA)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        rw = max(1, x2 - x1)
        rh = max(1, y2 - y1)
        r = max(8, min(rw, rh) // 2)
        r = min(r, 48)
        cv2.circle(frame_bgr, (cx, cy), r, _BALL_BGR, 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)

    def draw_detections(self, frame: np.ndarray, detections: sv.Detections, labels: list = None):
        """
        Annotate frame with boxes for ball/hoop.
        Players (class 4) are intentionally not drawn here.
        """
        annotated_frame = frame.copy()

        if detections is None or len(detections) == 0:
            return annotated_frame

        cls = detections.class_id
        if cls is None:
            return annotated_frame

        # Keep only non-player detections (ball/hoop/etc).
        other_mask = cls != 4
        other_detections = detections[other_mask]

        if len(other_detections) == 0:
            if labels:
                annotated_frame = self.label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections,
                    labels=labels,
                )
            return annotated_frame

        ocls = other_detections.class_id
        # Ball (0): explicit draw — small boxes are hard to see with default annotator colors.
        ball_mask = ocls == 0
        non_ball_mask = ~ball_mask

        if np.any(ball_mask):
            for bb in other_detections.xyxy[ball_mask].astype(int):
                self._draw_ball_bgr(annotated_frame, bb[0], bb[1], bb[2], bb[3])

        if np.any(non_ball_mask):
            rest = other_detections[non_ball_mask]
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame,
                detections=rest,
            )

        # Draw labels
        if labels:
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels,
            )

        return annotated_frame
