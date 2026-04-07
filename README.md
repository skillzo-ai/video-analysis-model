# 🧠 Skillzo Video Analysis (ML Submodule)

The **Skillzo Video Analysis** engine is the intelligence core of the platform. It implements a sophisticated computer vision pipeline that processes basketball game footage to detect players, track movement, and generate deep analytic statistics.

---

## 🔬 Pipeline Workflow

1. **Object Detection**: Uses **YOLOv11** to identify players, referees, the ball, and the hoops in every frame.
2. **Multi-Object Tracking**: Stable ID persistence using advanced tracking algorithms.
3. **Event Detection**: Heuristic and ML-based detectors for shots, passes, and makes/misses.
4. **Team Clustering**: Automatic assignment of players to teams based on visual jersey features.
5. **Data Export**: Generates a comprehensive JSON file containing play-by-play analytics.

---

## 🛠️ Stack & Dependencies

- **Framework**: Ultralytics YOLOv11
- **Deep Learning**: PyTorch
- **Tracking**: Supervision, ByteTrack
- **Image Processing**: OpenCV
- **Analysis**: Custom event detectors and clustering logic

---

## 🏃 Setup & Installation

### 1. Requirements
- Python 3.11+
- CUDA-capable GPU (Recommended for real-time performance)
- Model weights (`best.pt`)

### 2. Standalone Install
```bash
cd backend/video-analysis
uv sync
```

### 3. Model Weights
Place your trained YOLO weights as `best.pt` in this directory. The backend is configured to search for this file by default.

---

## 📟 CLI Usage
You can run the pipeline manually for testing:
```bash
# Detect and Track
uv run python main.py detect --source path/to/video.mp4 --model best.pt --output-folder ./output

# Team Clustering Test
uv run python main.py teams --image frame.jpg --bboxes boxes.json
```

---

## 📂 Internal Architecture

- `detection_pipeline/`: Orchestrates the flow from raw video to tracked data.
- `detectors/`: Specialized logic for basketball-specific events.
- `team_clustering/`: Vision-based team identification.
- `models/`: Configuration and model arch definitions.

---

## ⚖️ License
This module is part of the Skillzo project and is licensed under the **MIT License**.

© 2026 Skillzo Team. Released under the MIT License.
