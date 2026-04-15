"""
Upload image + YOLO label pairs produced by the main pipeline (no model inference here).

During ``VideoProcessor.process_video``, frames and ``.txt`` annotations are written under
``{output_folder}/{video_stem}/`` when Roboflow export is enabled. This module only reads that
folder and pushes files to Roboflow.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

_REPO_ROOT = Path(__file__).resolve().parent.parent

load_dotenv()


def _extract_class_name_from_line(line: str) -> str | None:
    """
    One class per line: either ``Ball`` or ``0 → Ball`` / ``4: Player`` (index + arrow copied from docs).

    Without this, lines like ``0 → Ball`` were stored as the full string and never matched ``Ball``.
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    # "0 → Ball", "0->Ball", "0 => Ball", "0: Ball", "0 = Ball", "0 - Ball"
    m = re.match(
        r"^\d+\s*(?:\u2192|→|->|=>|[:=]|\s-\s)\s*(.+)$",
        s,
    )
    if m:
        return m.group(1).strip()
    return s


def read_class_names(path: str | Path) -> list[str]:
    """One display name per line; ``#`` comments skipped; supports ``N → Name`` export from Roboflow UI."""
    text = Path(path).read_text(encoding="utf-8")
    out: list[str] = []
    for ln in text.splitlines():
        name = _extract_class_name_from_line(ln)
        if name:
            out.append(name)
    return out


def remap_yolo_label_to_roboflow(
    label_text: str,
    *,
    local_names: list[str],
    roboflow_names: list[str],
) -> str:
    """
    Map each box from local model class index → Roboflow index by **class name**.

    Roboflow "locked classes" require annotation IDs to match the project's class list order.
    Local exports use ``best.pt`` indices (e.g. Player=4) which often differ from Roboflow (e.g. Player=0).
    """
    name_to_rf = {n.casefold(): i for i, n in enumerate(roboflow_names)}
    out_lines: list[str] = []
    for line in label_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            local_id = int(parts[0])
        except ValueError:
            continue
        if local_id < 0 or local_id >= len(local_names):
            continue
        lname = local_names[local_id].strip()
        rid = name_to_rf.get(lname.casefold())
        if rid is None:
            continue
        out_lines.append(f"{rid} {' '.join(parts[1:])}")
    return "\n".join(out_lines) + ("\n" if out_lines else "")


def yolo_labelmap_dict(class_names: list[str]) -> dict[int, str]:
    """Roboflow expects ``{class_index: display_name}`` so YOLO ids map to locked project classes."""
    return {i: class_names[i] for i in range(len(class_names))}


def get_roboflow_project(
    api_key: str,
    workspace_id: str,
    project_id: str,
):
    """Single ``Roboflow`` → workspace → project (reuse for batch uploads)."""
    rf = Roboflow(api_key=api_key)
    return rf.workspace(workspace_id).project(project_id)


def upload_with_project(
    project,
    *,
    image_path: str | Path,
    annotation_path: str | Path,
    split: str = "train",
    annotation_labelmap: dict[int, str] | None = None,
) -> None:
    """Upload one image + label using an already-resolved ``project`` (no extra API client setup)."""
    project.upload(
        image_path=str(image_path),
        annotation_path=str(annotation_path),
        split=split,
        annotation_labelmap=annotation_labelmap,
    )


def upload_annotation(
    *,
    api_key: str,
    workspace_id: str,
    project_id: str,
    image_path: str | Path,
    annotation_path: str | Path,
    split: str = "train",
    annotation_labelmap: dict[int, str] | None = None,
) -> None:
    """One-shot upload (constructs client once). Prefer :func:`get_roboflow_project` + :func:`upload_with_project` in a loop."""
    project = get_roboflow_project(api_key, workspace_id, project_id)
    upload_with_project(
        project,
        image_path=image_path,
        annotation_path=annotation_path,
        split=split,
        annotation_labelmap=annotation_labelmap,
    )


def upload_export_folder_to_roboflow(
    export_dir: str | Path,
    *,
    roboflow_api_key: str | None = None,
    workspace_id: str = "kartiks-workspace-ia4hy",
    project_id: str = "basketball-players-arj24",
    upload_split: str = "train",
    max_uploads: int | None = None,
    roboflow_target_classes: str | Path | None = None,
) -> int:
    """
    Upload every ``images/*.jpg`` with a matching ``labels/{stem}.txt`` under ``export_dir``.

    If ``roboflow_target_classes`` is set (or ``{export_dir}/roboflow_target_classes.txt`` exists),
    labels are remapped from ``classes.txt`` (local model order) to Roboflow's class order **by name**,
    which fixes "locked annotation classes" rejections when indices differ between training weights
    and the Roboflow project.

    Returns the number of successful uploads.
    """
    import tempfile

    base = Path(export_dir).resolve()
    images_dir = base / "images"
    labels_dir = base / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    api_key = roboflow_api_key or os.getenv("ROBOFLOW_API")
    if not api_key:
        raise ValueError("ROBOFLOW_API is not set.")

    target_path: Path | None = None
    if roboflow_target_classes:
        target_path = Path(roboflow_target_classes).resolve()
    else:
        cand = base / "roboflow_target_classes.txt"
        if cand.is_file():
            target_path = cand

    local_names: list[str] = []
    roboflow_names: list[str] | None = None
    if target_path is not None:
        if not target_path.is_file():
            raise FileNotFoundError(f"Roboflow class list not found: {target_path}")
        local_classes_path = base / "classes.txt"
        if not local_classes_path.is_file():
            raise FileNotFoundError(
                f"Remapping requires {local_classes_path} (local model class names by index)."
            )
        local_names = read_class_names(local_classes_path)
        roboflow_names = read_class_names(target_path)
        print(
            f"Remapping labels by name: {len(local_names)} local classes → "
            f"{len(roboflow_names)} Roboflow classes (from {target_path.name})"
        )
    else:
        print(
            "Warning: no roboflow_target_classes.txt — uploading local YOLO class indices as-is. "
            "If uploads are rejected for locked classes, add a text file listing Roboflow class "
            "names in project order (see Dataset/roboflow_target_classes.example.txt)."
        )
        lc = base / "classes.txt"
        if lc.is_file():
            local_names = read_class_names(lc)

    # Labelmap tells the API what each YOLO class index means (required for many locked-class projects).
    if roboflow_names is not None:
        labelmap_for_upload = yolo_labelmap_dict(roboflow_names)
    elif local_names:
        labelmap_for_upload = yolo_labelmap_dict(local_names)
    else:
        labelmap_for_upload = None
        print(
            "Warning: no labelmap (missing classes.txt) — Roboflow may reject annotations on locked classes."
        )

    jpgs = sorted(images_dir.glob("*.jpg"))
    if not jpgs:
        jpgs = sorted(images_dir.glob("*.jpeg"))
    if not jpgs:
        jpgs = sorted(images_dir.glob("*.png"))

    if not jpgs:
        print(f"No images under {images_dir}; nothing to upload.")
        return 0

    project = get_roboflow_project(api_key, workspace_id, project_id)
    count = 0
    for img_path in jpgs:
        if max_uploads is not None and count >= max_uploads:
            break
        stem = img_path.stem
        lbl_path = labels_dir / f"{stem}.txt"
        if not lbl_path.is_file():
            print(f"Skip (no label): {img_path.name}")
            continue

        raw_txt = lbl_path.read_text(encoding="utf-8")
        upload_path: str | Path = lbl_path
        tmp_path: str | None = None
        if roboflow_names is not None:
            mapped = remap_yolo_label_to_roboflow(
                raw_txt, local_names=local_names, roboflow_names=roboflow_names
            )
            if not mapped.strip():
                print(f"Skip (no boxes after remap): {img_path.name}")
                continue
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                delete=False,
                encoding="utf-8",
            )
            try:
                tmp.write(mapped)
                tmp.close()
                upload_path = tmp.name
                tmp_path = tmp.name
            except Exception:
                Path(tmp.name).unlink(missing_ok=True)
                raise

        try:
            upload_with_project(
                project,
                image_path=img_path,
                annotation_path=upload_path,
                split=upload_split,
                annotation_labelmap=labelmap_for_upload,
            )
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)

        count += 1
        print(f"Uploaded {stem} ({count})")

    print(f"Roboflow upload finished: {count} image(s) from {base}")
    return count


def upload_annonation(image_path=None, annonation_path=None, **kwargs):
    """Legacy helper: upload a single image + YOLO label."""
    api_key = os.getenv("ROBOFLOW_API")
    if not api_key:
        raise ValueError("ROBOFLOW_API is not set.")
    upload_annotation(
        api_key=api_key,
        workspace_id=kwargs.get("workspace_id", "kartiks-workspace-ia4hy"),
        project_id=kwargs.get("project_id", "basketball-players-arj24"),
        image_path=image_path,
        annotation_path=annonation_path,
        split=kwargs.get("split", "train"),
        annotation_labelmap=kwargs.get("annotation_labelmap"),
    )


def extract_frames(video_path=None, interval: int = 25, output_dir: str | None = None):
    """Extract raw frames every ``interval`` (OpenCV only; no model)."""
    import cv2

    if not video_path:
        return
    out = Path(output_dir or "images")
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            cv2.imwrite(str(out / f"frame_{frame_count}.jpg"), frame)
        frame_count += 1
    cap.release()
    print(out.resolve())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload a dataset folder (images/ + labels/) to Roboflow — no model runs here."
    )
    p.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Export root containing images/ and labels/ (e.g. output/my_video/)",
    )
    p.add_argument(
        "--workspace",
        type=str,
        default="kartiks-workspace-ia4hy",
        help="Roboflow workspace id",
    )
    p.add_argument(
        "--project",
        type=str,
        default="basketball-players-arj24",
        help="Roboflow project id",
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "valid", "test"),
        help="Roboflow split",
    )
    p.add_argument(
        "--max-uploads",
        type=int,
        default=None,
        help="Stop after this many uploads",
    )
    p.add_argument(
        "--roboflow-target-classes",
        type=str,
        default=None,
        help="Text file: one Roboflow class name per line (project order). Enables index remapping.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    upload_export_folder_to_roboflow(
        args.folder,
        workspace_id=args.workspace,
        project_id=args.project,
        upload_split=args.split,
        max_uploads=args.max_uploads,
        roboflow_target_classes=args.roboflow_target_classes,
    )


if __name__ == "__main__":
    main()
