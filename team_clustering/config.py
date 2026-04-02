from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple


ColorSpace = Literal["LAB", "HSV"]


@dataclass(frozen=True)
class TeamClusteringConfig:
    """
    Configuration for jersey-color based two-team clustering.
    """

    # --- Cropping (torso focus) ---
    torso_y0: float = 0.20  # fraction from bbox top
    torso_y1: float = 0.75  # fraction from bbox top
    torso_x_pad: float = 0.10  # crop narrower to avoid background

    # --- Preprocess / performance ---
    crop_resize_max_side: int = 96

    # --- Color extraction ---
    color_space: ColorSpace = "LAB"
    hsv_sat_min: int = 40
    hsv_val_min: int = 35
    # Exclude green-ish pixels (typical grass) in HSV hue space.
    # OpenCV hue range is [0, 179]. Set to None to disable.
    hsv_exclude_green_hue_lo: int | None = 35
    hsv_exclude_green_hue_hi: int | None = 95

    # If too few masked pixels remain, fall back to using the full crop.
    min_masked_fraction: float = 0.05

    # Optional: refine "representative jersey color" by clustering pixels in the crop.
    per_crop_kmeans_k: int | None = None  # e.g. 2 or 3; None disables
    per_crop_kmeans_max_iter: int = 20

    # --- Team clustering (players) ---
    global_k: int = 2
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 200
    kmeans_random_state: int = 0

    # --- Temporal smoothing (video) ---
    temporal_smoothing: bool = True
    temporal_vote_decay: float = 0.0  # 0 disables decay; e.g. 0.02 slowly forgets old votes

    # --- Visualization (ellipse) ---
    ellipse_color_team_a_bgr: Tuple[int, int, int] = (255, 0, 0)  # Blue
    ellipse_color_team_b_bgr: Tuple[int, int, int] = (0, 0, 255)  # Red
    possession_highlight_bgr: Tuple[int, int, int] = (0, 255, 255)  # Yellow/Cyan highlight
    ellipse_thickness: int = 2
    ellipse_axis_w_scale: float = 0.50  # axes x = w * scale
    ellipse_axis_h_scale: float = 0.15  # axes y = h * scale

    # Optional label drawing
    draw_bbox: bool = False
    draw_text: bool = True
    text_scale: float = 0.5
    text_thickness: int = 1

    # --- Debug ---
    debug: bool = False
    debug_dir: str = "debug/team_clustering"
    debug_save_crops: bool = True
    debug_save_masks: bool = True
    debug_verbose: bool = False

    def debug_path(self) -> Path:
        return Path(self.debug_dir)

