# Skillzo Video Analysis (ML submodule)

This repository is a **Git submodule** mounted at `backend/video-analysis/` inside the Skillzo backend. It implements the computer-vision pipeline: **object detection / tracking**, **event detectors** (pass, shot, make/miss), and **team statistics** exported as JSON. The FastAPI backend imports `run_detection_pipeline` from this package’s `main.py` and does not need to duplicate ML code.

**Upstream remote** is configured in the parent backend’s `.gitmodules` (typically `video-analysis-model`). After moving the project to a new org, update that URL and run `git submodule sync`.

---

## Role in the full system

1. User uploads a video through the frontend → **FastAPI** saves it and enqueues processing.
2. Backend calls `run_detection_pipeline(local_video_path, model_path=..., output_folder=...)` defined in this repo’s `main.py`.
3. Pipeline writes a **tracked video** and a **stats JSON**; backend transcodes video to H.264, uploads to Supabase **analysis** bucket, and updates the `videos` table.

If this submodule is not checked out or `best.pt` is missing, the backend’s import may fall back to a no-op pipeline and uploads will fail analysis.

---

## Prerequisites

- **Python 3.11+**
- **uv** or **pip**
- Optional: **CUDA**-capable GPU for reasonable inference speed  
  Default `pyproject.toml` uses PyTorch **CUDA 12.1** wheels. For CPU-only machines, replace the `torch` / `torchvision` source index with CPU builds.

---

## Install (standalone development)

```bash
cd video-analysis   # from backend: backend/video-analysis
uv sync
```

If you do not use `uv`, create a virtual environment and install dependencies from `pyproject.toml` with your preferred tool (for example `pip install .` from this directory, depending on your packaging setup).

Copy `env.example` to `.env` if you use **Roboflow** or other secrets from environment variables inside detectors or training scripts.

```env
ROBOFLOW_API=your_key_if_needed
```

---

## Default model weights

The backend expects a **`best.pt`** file in this directory (see `video_analysis_router.py` in the backend). Additional weights such as `yolov8n.pt` / `yolo26n.pt` may exist for experiments. Large `.pt` files are often excluded from Git or stored with **Git LFS** — confirm with your team when transferring the project.

---

## Directories (runtime)

The backend ensures these exist under `video-analysis/`:

| Directory | Use |
|-----------|-----|
| `input_videos/` | Staged uploads before / during processing |
| `temp_analysis/` | Intermediate pipeline output |
| `outputs_videos/` | Final local outputs before upload to Supabase |

Other folders are part of the codebase:

| Folder | Use |
|--------|-----|
| `detection_pipeline/` | Core video processing / tracking orchestration |
| `detectors/` | Pass, shot, make/miss logic |
| `team_clustering/` | Jersey-color–style team separation helpers |
| `models/`, `config/` | Model and configuration assets |
| `Training/`, `Dataset/` | Training and data (may be large) |

---

## Public API used by the backend

### `run_detection_pipeline(...)`

Defined in `main.py`. Signature (summary):

- `source`: path to input video  
- `model_path`: defaults to `"best.pt"`  
- `output_folder`: directory for `{stem}_tracked.mp4` and `{stem}.json`  
- `output_path`: legacy single-file video path (optional)  
- `log_events_all_frames`: verbose per-frame logging if `True`  

Returns a `dict` with at least `"video"` and `"stats_json"` string paths.

### `run_team_clustering_on_image(...)`

Utility for clustering teams on a single frame + bbox JSON (CLI `teams` subcommand).

---

## Command-line usage (local tests)

From this directory, with the virtual environment active:

```bash
# Full detection + tracking + stats JSON
uv run python main.py detect --source path/to/game.mp4 --model best.pt --output-folder ./output

# Team clustering on one frame
uv run python main.py teams --image frame.jpg --bboxes boxes.json --output annotated.jpg
```

---

## Training and datasets

`Training/` and `Dataset/` support model development. Training is **not** required to run the website if you already have compatible `best.pt` weights.

---

## Handoff checklist for a new team

- [ ] Submodule checked out: `git submodule update --init --recursive` from monorepo root  
- [ ] `best.pt` (or agreed model) present and path matches backend configuration  
- [ ] Python deps install cleanly on lab machines (adjust Torch CUDA/CPU as needed)  
- [ ] `.env` populated if Roboflow or other APIs are required  
- [ ] Read [../README.md](../README.md) (backend) for how uploads trigger this pipeline  

Parent monorepo overview: [../../README.md](../../README.md).
