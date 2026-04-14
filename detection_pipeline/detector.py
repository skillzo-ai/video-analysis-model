import numpy as np
from ultralytics import YOLO
import supervision as sv


def _merge_detections(a: sv.Detections, b: sv.Detections) -> sv.Detections:
    """Concatenate two supervision Detections (pre-tracker); pads optional fields."""
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a

    xyxy = np.concatenate([a.xyxy, b.xyxy], axis=0)

    confidence = None
    if a.confidence is not None or b.confidence is not None:
        ad = a.confidence if a.confidence is not None else np.zeros((len(a),), dtype=np.float32)
        bd = b.confidence if b.confidence is not None else np.zeros((len(b),), dtype=ad.dtype)
        confidence = np.concatenate([ad, bd], axis=0)

    class_id = None
    if a.class_id is not None or b.class_id is not None:
        ad = a.class_id if a.class_id is not None else np.zeros((len(a),), dtype=np.int32)
        bd = b.class_id if b.class_id is not None else np.zeros((len(b),), dtype=ad.dtype)
        class_id = np.concatenate([ad, bd], axis=0)

    tracker_id = None
    if a.tracker_id is not None or b.tracker_id is not None:
        ad = (
            a.tracker_id
            if a.tracker_id is not None
            else np.full((len(a),), -1, dtype=np.int32)
        )
        bd = (
            b.tracker_id
            if b.tracker_id is not None
            else np.full((len(b),), -1, dtype=ad.dtype)
        )
        tracker_id = np.concatenate([ad, bd], axis=0)

    mask = None
    if getattr(a, "mask", None) is not None or getattr(b, "mask", None) is not None:
        am = a.mask if getattr(a, "mask", None) is not None else None
        bm = b.mask if getattr(b, "mask", None) is not None else None
        if am is not None and bm is not None:
            mask = np.concatenate([am, bm], axis=0)
        elif am is not None:
            mask = am
        else:
            mask = bm

    return sv.Detections(
        xyxy=xyxy,
        mask=mask,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id,
    )


class Detector:
    """
    Main YOLO weights only (players, hoops, etc.). Ball class is not used here — ball comes from
    :class:`BallTracker` + ``ball_detector_model.pt`` in :class:`VideoProcessor`.
    """

    def __init__(self, model_path: str = "best.pt"):
        self.main_model = YOLO(model_path)

        self.keep_classes = [0, 2, 4]
        self.keep_classes_non_ball = [c for c in self.keep_classes if c != 0]
        self.player_hoop_classes = [2, 4]

    def get_detections(
        self,
        frame: np.ndarray,
        conf: float = 0.15,
        iou: float = 0.5,
        player_conf: float = 0.3,
    ):
        """
        Single-frame main model: hoop + player (no ball). Stricter conf for players/hoops.
        """
        main_results = self.main_model(
            frame,
            conf=conf,
            iou=iou,
            imgsz=832,
            verbose=False,
            classes=self.keep_classes_non_ball,
        )[0]
        det_main = sv.Detections.from_ultralytics(main_results)

        if len(det_main) == 0:
            return det_main

        keep_mask = np.isin(det_main.class_id, self.keep_classes_non_ball)
        det_main = det_main[keep_mask]
        if det_main.confidence is not None:
            is_player_or_hoop = np.isin(det_main.class_id, self.player_hoop_classes)
            hi_ok = (~is_player_or_hoop) | (det_main.confidence >= float(player_conf))
            det_main = det_main[hi_ok]

        return det_main
