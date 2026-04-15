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
from team_clustering.temporal_team_classification import TemporalTeamConfig
from tracking import TrackingConfig


BASE_DIR = Path(__file__).resolve().parent


def _resolve_existing_path(path_str: str, *, base_dir: Path = BASE_DIR) -> Path:
    """
    Resolve a path from the current working directory first, then relative to this
    package's directory so model files can be found no matter where the process starts.
    """
    candidate = Path(path_str)
    if candidate.exists():
        return candidate.resolve()

    fallback = (base_dir / candidate).resolve()
    if fallback.exists():
        return fallback

    raise FileNotFoundError(path_str)


def run_detection_pipeline(
    source: str,
    model_path: str = "best.pt",
    output_path: str | None = None,
    *,
    ball_model_path: str = "ball_detector_model.pt",
    ball_read_stub: bool = False,
    ball_stub_path: str | None = None,
    output_folder: str | None = None,
    log_events_all_frames: bool = False,
    tracking_config: TrackingConfig | dict | None = None,
    temporal_team_config: TemporalTeamConfig | dict | None = None,
    roboflow_export: bool = False,
    roboflow_upload: bool = False,
    roboflow_interval: int = 25,
    roboflow_out_dir: str | None = None,
    roboflow_max_uploads: int | None = None,
    roboflow_workspace: str = "kartiks-workspace-ia4hy",
    roboflow_project: str = "basketball-players-arj24",
    roboflow_split: str = "train",
    roboflow_target_classes: str | None = None,
) -> dict[str, str]:
    """
    Run tracking on `source` and write:
      - Tracked video: ``{input_stem}_tracked.mp4``
      - Stats JSON: ``{input_stem}.json`` with team_A / team_B counts and ``track_teams`` (ByteTrack id → Team A/B).

    If ``output_folder`` is set, both files go under that directory (created if needed).
    If only ``output_path`` is set (legacy), that path is used for the video and the JSON
    is written next to it as ``{input_stem}.json``.
    If neither is set, defaults to folder ``output`` under the current working directory.

    If ``roboflow_export`` is True, during ``process_video`` the pipeline also writes raw frames and
    YOLO ``.txt`` labels every ``roboflow_interval`` frames under ``{output}/{video_stem}/``.
    If ``roboflow_upload`` is True, those files are uploaded from that folder after processing
    (see ``Dataset.upload_annonations.upload_export_folder_to_roboflow`` — no second model pass).
    For Roboflow projects with locked classes, pass ``roboflow_target_classes`` (or place
    ``roboflow_target_classes.txt`` in the export folder) listing class names in Roboflow's order
    so label indices are remapped by name from ``classes.txt``.
    """
    try:
        model_path = str(_resolve_existing_path(model_path))
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        ball_model_path = str(_resolve_existing_path(ball_model_path))
    except FileNotFoundError:
        raise FileNotFoundError(f"Ball model file not found: {ball_model_path}")

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

    tc = (
        TrackingConfig.from_dict(tracking_config)
        if isinstance(tracking_config, dict)
        else tracking_config
    )
    tteam = (
        TemporalTeamConfig.from_dict(temporal_team_config)
        if isinstance(temporal_team_config, dict)
        else temporal_team_config
    )

    export_dir_for_processor: str | None = None
    if roboflow_export:
        export_dir_for_processor = (
            str(Path(roboflow_out_dir).resolve())
            if roboflow_out_dir
            else str((video_out.parent / input_stem).resolve())
        )

    processor = VideoProcessor(
        model_path=model_path,
        ball_model_path=ball_model_path,
        ball_read_stub=ball_read_stub,
        ball_stub_path=ball_stub_path,
        output_path=str(video_out),
        pass_detector=pass_detector,
        shot_detector=shot_detector,
        make_miss_detector=make_miss_detector,
        log_events_all_frames=log_events_all_frames,
        tracking_config=tc,
        temporal_team_config=tteam,
        export_dataset_dir=export_dir_for_processor,
        export_interval=roboflow_interval,
    )
    processor.process_video(source=source)

    stats = processor.team_stats_export_dict()
    stats["track_teams"] = processor.export_team_track_json()
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Tracking complete. Video: {video_out}")
    print(f"Stats JSON: {json_out}")

    result: dict[str, str] = {"video": str(video_out), "stats_json": str(json_out)}

    if roboflow_export and export_dir_for_processor:
        print(
            f"Dataset export (during tracking): every {roboflow_interval} frames → {export_dir_for_processor}"
        )
        result["roboflow_export_dir"] = export_dir_for_processor

    if roboflow_export and roboflow_upload and export_dir_for_processor:
        from Dataset.upload_annonations import upload_export_folder_to_roboflow

        rtc_path: str | None = None
        if roboflow_target_classes:
            try:
                rtc_path = str(_resolve_existing_path(roboflow_target_classes))
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Roboflow target classes file not found: {roboflow_target_classes}"
                ) from e

        upload_export_folder_to_roboflow(
            export_dir_for_processor,
            workspace_id=roboflow_workspace,
            project_id=roboflow_project,
            upload_split=roboflow_split,
            max_uploads=roboflow_max_uploads,
            roboflow_target_classes=rtc_path,
        )

    return result


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
    detect.add_argument("--model", type=str, default="best.pt", help="Path to main YOLO weights (players, hoops; ball class ignored)")
    detect.add_argument(
        "--ball-model",
        type=str,
        default="ball_detector_model.pt",
        help="Path to ball-only YOLO weights (BallTracker batch pipeline)",
    )
    detect.add_argument(
        "--ball-stub",
        type=str,
        default=None,
        help="Optional pickle path to cache ball tracks (load with --read-ball-stub, always saved after run)",
    )
    detect.add_argument(
        "--read-ball-stub",
        action="store_true",
        help="Load ball tracks from --ball-stub if length matches video frame count",
    )
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
    detect.add_argument(
        "--no-roboflow",
        action="store_true",
        help="Skip Roboflow export/upload after tracking (default: export runs at the end)",
    )
    detect.add_argument(
        "--roboflow-upload",
        action="store_true",
        help="After export, also upload each image+label to Roboflow (needs ROBOFLOW_API)",
    )
    detect.add_argument(
        "--roboflow-interval",
        type=int,
        default=25,
        help="Frame interval for Roboflow export (default: 25)",
    )
    detect.add_argument(
        "--roboflow-out",
        type=str,
        default=None,
        help="Export root folder named like the video (default: {output_folder}/{video_stem}/)",
    )
    detect.add_argument(
        "--roboflow-max-uploads",
        type=int,
        default=None,
        help="Stop Roboflow upload after this many images (testing)",
    )
    detect.add_argument(
        "--roboflow-workspace",
        type=str,
        default="kartiks-workspace-ia4hy",
        help="Roboflow workspace id for uploads",
    )
    detect.add_argument(
        "--roboflow-project",
        type=str,
        default="basketball-players-arj24",
        help="Roboflow project id for uploads",
    )
    detect.add_argument(
        "--roboflow-split",
        type=str,
        default="train",
        choices=("train", "valid", "test"),
        help="Roboflow split for uploads",
    )
    detect.add_argument(
        "--roboflow-target-classes",
        type=str,
        default=None,
        help=(
            "File: one Roboflow class name per line (project order). "
            "Remaps local YOLO ids by name so locked-class uploads succeed. "
            "Or put roboflow_target_classes.txt inside the export folder."
        ),
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
                ball_model_path=args.ball_model,
                ball_read_stub=bool(args.read_ball_stub),
                ball_stub_path=args.ball_stub,
                output_path=legacy_video,
                output_folder=out_folder,
                log_events_all_frames=bool(args.log_events_all_frames),
                roboflow_export=not bool(args.no_roboflow),
                roboflow_upload=bool(args.roboflow_upload),
                roboflow_interval=int(args.roboflow_interval),
                roboflow_out_dir=args.roboflow_out,
                roboflow_max_uploads=args.roboflow_max_uploads,
                roboflow_workspace=args.roboflow_workspace,
                roboflow_project=args.roboflow_project,
                roboflow_split=args.roboflow_split,
                roboflow_target_classes=args.roboflow_target_classes,
            )
        elif args.command == "teams":
            run_team_clustering_on_image(args.image, args.bboxes, args.output, debug=bool(args.debug))
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
