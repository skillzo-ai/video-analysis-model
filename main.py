import argparse
import json
import os
from pathlib import Path

import cv2

from detection_pipeline.processor import VideoProcessor
from team_clustering.config import TeamClusteringConfig
from team_clustering.pipeline import classify_teams

def run_detection_pipeline(source: str, model_path: str = "best.pt", output_path: str = "output_tracked.mp4"):
    """
    Function to be called from FastAPI or other modules.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source video not found: {source}")

    processor = VideoProcessor(
        model_path=model_path,
        output_path=output_path
    )
    # Team clustering is integrated inside VideoProcessor (player ellipses).
    processor.process_video(source=source)
    return output_path


def run_team_clustering_on_image(
    image_path: str,
    bboxes_json_path: str,
    output_path: str = "team_clustering_annotated.jpg",
    *,
    debug: bool = False,
):
    """
    Run jersey-color clustering on a single image.

    Expects `bboxes_json_path` to be a JSON file containing a list of bboxes:
      [[x, y, w, h], [x, y, w, h], ...]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(bboxes_json_path):
        raise FileNotFoundError(f"BBoxes JSON not found: {bboxes_json_path}")

    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Failed to read image: {image_path}")

    with open(bboxes_json_path, "r", encoding="utf-8") as f:
        bboxes = json.load(f)
    if not isinstance(bboxes, list):
        raise ValueError("BBoxes JSON must be a list of [x,y,w,h] boxes.")

    cfg = TeamClusteringConfig(debug=debug, draw_text=True)
    result = classify_teams(frame, bboxes, cfg)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), result["annotated_frame"])
    if not ok:
        raise ValueError(f"Failed to write output image: {output_path}")

    print("labels:", result["labels"])
    print("teams:", result["teams"])
    print("cluster_centers:", result["cluster_centers"])
    print("saved:", str(out_path))
    return str(out_path)

def main():
    parser = argparse.ArgumentParser(description="Basketball Detection/Tracking + Team Clustering")
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="Run detection + tracking on a video")
    detect.add_argument("--source", type=str, required=True, help="Path to input video")
    detect.add_argument("--model", type=str, default="best.pt", help="Path to model weights")
    detect.add_argument("--output", type=str, default="output_tracked.mp4", help="Path to output video")

    teams = sub.add_parser("teams", help="Run team clustering on a single image + bboxes JSON")
    teams.add_argument("--image", type=str, required=True, help="Path to input image (frame)")
    teams.add_argument("--bboxes", type=str, required=True, help="Path to JSON file with [[x,y,w,h], ...]")
    teams.add_argument("--output", type=str, default="team_clustering_annotated.jpg", help="Path to output annotated image")
    teams.add_argument("--debug", action="store_true", help="Enable debug outputs (crops/masks + centers logging)")

    args = parser.parse_args()

    try:
        if args.command == "detect":
            run_detection_pipeline(source=args.source, model_path=args.model, output_path=args.output)
        elif args.command == "teams":
            run_team_clustering_on_image(args.image, args.bboxes, args.output, debug=bool(args.debug))
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
