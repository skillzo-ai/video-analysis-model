import argparse
import json
import os
from pathlib import Path

import cv2

from detectors.make_miss_detector import MakeMissDetector
from detectors.pass_detector import PassDetector
from detectors.shot_detector import ShotDetector
from detection_pipeline.processor import VideoProcessor
from team_clustering.config import TeamClusteringConfig
from team_clustering.pipeline import classify_teams

def run_detection_pipeline(
    source: str,
    model_path: str = "best.pt",
    output_path: str | None = None,
    *,
    output_folder: str | None = None,
    log_events_all_frames: bool = False,
) -> dict[str, str]:
    """
    Run tracking on `source` and write:
      - Tracked video: ``{input_stem}_tracked.mp4``
      - Stats JSON: ``{input_stem}.json`` with team_A / team_B counts.

    If ``output_folder`` is set, both files go under that directory (created if needed).
    If only ``output_path`` is set (legacy), that path is used for the video and the JSON
    is written next to it as ``{input_stem}.json``.
    If neither is set, defaults to folder ``output`` under the current working directory.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(source):
        raise FileNotFoundError(f"Source video not found: {source}")

    source_path = Path(source)
    input_stem = source_path.stem

    if output_folder is not None:
        out_dir = Path(output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        video_out = out_dir / f"{input_stem}_tracked.mp4"
        json_out = out_dir / f"{input_stem}.json"
    elif output_path is not None:
        video_out = Path(output_path)
        video_out.parent.mkdir(parents=True, exist_ok=True)
        json_out = video_out.parent / f"{input_stem}.json"
    else:
        out_dir = Path("output")
        out_dir.mkdir(parents=True, exist_ok=True)
        video_out = out_dir / f"{input_stem}_tracked.mp4"
        json_out = out_dir / f"{input_stem}.json"

    pass_detector = PassDetector()
    shot_detector = ShotDetector()
    make_miss_detector = MakeMissDetector()

    processor = VideoProcessor(
        model_path=model_path,
        output_path=str(video_out),
        pass_detector=pass_detector,
        shot_detector=shot_detector,
        make_miss_detector=make_miss_detector,
        log_events_all_frames=log_events_all_frames,
    )
    processor.process_video(source=source)

    stats = processor.team_stats_export_dict()
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Tracking complete. Video: {video_out}")
    print(f"Stats JSON: {json_out}")
    return {"video": str(video_out), "stats_json": str(json_out)}


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
    detect.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Directory for {input_name}_tracked.mp4 and {input_name}.json (default: ./output)",
    )
    detect.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit path for tracked video only; stats JSON is written beside it as {input_stem}.json",
    )
    detect.add_argument(
        "--log-events-all-frames",
        action="store_true",
        help="Print pass/shot/make JSON every frame (default: only when any event is True)",
    )

    teams = sub.add_parser("teams", help="Run team clustering on a single image + bboxes JSON")
    teams.add_argument("--image", type=str, required=True, help="Path to input image (frame)")
    teams.add_argument("--bboxes", type=str, required=True, help="Path to JSON file with [[x,y,w,h], ...]")
    teams.add_argument("--output", type=str, default="team_clustering_annotated.jpg", help="Path to output annotated image")
    teams.add_argument("--debug", action="store_true", help="Enable debug outputs (crops/masks + centers logging)")

    args = parser.parse_args()

    try:
        if args.command == "detect":
            out_folder = args.output_folder
            if out_folder is None and args.output is None:
                out_folder = "output"
            # If --output-folder is set, it controls both files; --output is ignored.
            legacy_video = args.output if args.output_folder is None else None
            run_detection_pipeline(
                source=args.source,
                model_path=args.model,
                output_path=legacy_video,
                output_folder=out_folder,
                log_events_all_frames=bool(args.log_events_all_frames),
            )
        elif args.command == "teams":
            run_team_clustering_on_image(args.image, args.bboxes, args.output, debug=bool(args.debug))
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
