"""
Upload a Roboflow export folder (images + YOLO labels) produced by ``main.py detect``.

Run after detection, e.g.::

    python main.py detect --source input_videos/clip.mp4
    python upload_to_roboflow.py --export-dir output/clip

Requires ``ROBOFLOW_API`` (or ``ROBOFLOW_KEY``) in the environment and ``Dataset/upload_annonations.py``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _resolve_existing_path(path_str: str, *, base_dir: Path = BASE_DIR) -> Path:
    candidate = Path(path_str)
    if candidate.exists():
        return candidate.resolve()
    fallback = (base_dir / candidate).resolve()
    if fallback.exists():
        return fallback
    raise FileNotFoundError(path_str)


def run_roboflow_upload(
    export_dir: str,
    *,
    workspace_id: str | None = None,
    project_id: str | None = None,
    upload_split: str | None = None,
    max_uploads: int | None = None,
    roboflow_target_classes: str | None = None,
) -> None:
    """Upload ``export_dir`` to Roboflow using ``Dataset.upload_annonations``."""
    from Dataset.upload_annonations import upload_export_folder_to_roboflow

    workspace_id = workspace_id or os.environ.get("ROBOFLOW_WORKSPACE", "kartiks-workspace-ia4hy")
    project_id = project_id or os.environ.get("ROBOFLOW_PROJECT", "basketball-players-arj24")
    upload_split = upload_split or os.environ.get("ROBOFLOW_SPLIT", "train")
    if max_uploads is None and (m := os.environ.get("ROBOFLOW_MAX_UPLOADS", "").strip()):
        max_uploads = int(m)

    rtc_path: str | None = None
    if roboflow_target_classes:
        rtc_path = str(_resolve_existing_path(roboflow_target_classes))
    else:
        default = BASE_DIR / "Dataset" / "roboflow_target_classes.txt"
        if default.is_file():
            rtc_path = str(default)

    upload_export_folder_to_roboflow(
        export_dir,
        workspace_id=workspace_id,
        project_id=project_id,
        upload_split=upload_split,
        max_uploads=max_uploads,
        roboflow_target_classes=rtc_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a detection export folder to Roboflow (after main.py detect)."
    )
    parser.add_argument(
        "--export-dir",
        required=True,
        help="Folder with frames + YOLO labels (same as roboflow_export_dir from detection).",
    )
    parser.add_argument("--workspace", default=None, help="Override ROBOFLOW_WORKSPACE")
    parser.add_argument("--project", default=None, help="Override ROBOFLOW_PROJECT")
    parser.add_argument(
        "--split",
        default=None,
        choices=("train", "valid", "test"),
        help="Override ROBOFLOW_SPLIT",
    )
    parser.add_argument("--max-uploads", type=int, default=None)
    parser.add_argument(
        "--target-classes",
        default=None,
        help="Path to roboflow_target_classes.txt (default: Dataset/roboflow_target_classes.txt if present)",
    )
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    if not export_dir.is_dir():
        print(f"Error: not a directory: {export_dir}", file=sys.stderr)
        sys.exit(1)

    api = (os.environ.get("ROBOFLOW_API") or os.environ.get("ROBOFLOW_KEY") or "").strip()
    if not api:
        print("Error: set ROBOFLOW_API or ROBOFLOW_KEY", file=sys.stderr)
        sys.exit(1)

    run_roboflow_upload(
        str(export_dir.resolve()),
        workspace_id=args.workspace,
        project_id=args.project,
        upload_split=args.split,
        max_uploads=args.max_uploads,
        roboflow_target_classes=args.target_classes,
    )
    print("Roboflow upload finished.")


if __name__ == "__main__":
    main()
