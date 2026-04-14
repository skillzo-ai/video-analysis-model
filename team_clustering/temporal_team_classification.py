"""
Temporal two-team classification from jersey color.

Pipeline per frame:
  1. Crop torso band (default: 20%–60% of player bbox height) from BGR frame.
  2. Convert crop to HSV; run KMeans (k=2) on pixels → dominant jersey color.
  3. KMeans (k=2) on *unlocked* players' dominant colors → team labels (locked tracks skip re-detection).
  4. Align cluster ids to stable Team A/B using EMA reference colors.
  5. By default (``freeze_team_after_assignment``): each track is classified once when it first
     appears; its team never changes. Only **new** track IDs run jersey extraction + clustering.
     Optional legacy mode: majority vote over last N frames without freezing.

Designed for use with YOLO + ByteTrack outputs (xyxy boxes + stable track IDs).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields

import cv2
import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# --- Team label constants (string API) ---
TEAM_A = "Team A"
TEAM_B = "Team B"


@dataclass
class TemporalTeamConfig:
    """Tuning for jersey extraction, gating, clustering, and temporal smoothing."""

    # Torso band as fraction of full bbox height (top → bottom).
    jersey_y0: float = 0.20
    jersey_y1: float = 0.60
    # Horizontal inset as fraction of bbox width (reduces background at sides).
    torso_x_pad: float = 0.08

    # Ignore tiny players (area in pixels) or low-confidence detections.
    min_bbox_area_px: float = 400.0
    min_confidence: float = 0.25

    # Dominant color: KMeans on HSV pixels inside jersey crop.
    per_crop_kmeans_k: int = 2
    per_crop_max_pixels: int = 8000
    per_crop_min_pixels: int = 30

    # Frame-level team clustering on dominant colors (HSV, row-wise normalized for distance).
    frame_kmeans_n_init: int = 10
    frame_kmeans_max_iter: int = 200
    frame_kmeans_random_state: int = 42

    # Temporal: majority vote over last N *successful* frame predictions per track.
    history_length: int = 10

    # Stabilize Team A vs B assignment when k-means swaps cluster indices across frames.
    ref_center_momentum: float = 0.15  # EMA update rate for reference HSV centers (normalized space)

    # Once a track gets a team, never re-run jersey color or change team (only new ByteTrack IDs classify).
    freeze_team_after_assignment: bool = True

    # Calibration window: accumulate per-frame team votes (0/1) for this many seconds, then
    # majority-vote and lock. Set to 0 to disable (use immediate freeze / legacy mode instead).
    calibration_duration_sec: float = 20.0

    # BGR visualization (matches draw_team_bboxes defaults)
    color_team_a_bgr: tuple[int, int, int] = (255, 120, 0)
    color_team_b_bgr: tuple[int, int, int] = (60, 180, 255)
    possession_highlight_bgr: tuple[int, int, int] = (0, 255, 255)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TemporalTeamConfig:
        known = {f.name for f in fields(cls)}
        raw = {k: v for k, v in d.items() if k in known}
        for key in ("color_team_a_bgr", "color_team_b_bgr", "possession_highlight_bgr"):
            if key in raw and isinstance(raw[key], list):
                raw[key] = tuple(int(x) for x in raw[key])
        return cls(**raw)


def extract_jersey_crop(
    frame_bgr: np.ndarray,
    xyxy: np.ndarray | tuple[float, float, float, float],
    cfg: TemporalTeamConfig | None = None,
) -> np.ndarray | None:
    """
    Extract the jersey/torso band from a player bounding box (xyxy format).

    Uses only the vertical slice between ``jersey_y0`` and ``jersey_y1`` of the bbox height
    so shoes, court, and much of the head/background are excluded.

    Returns:
        BGR crop or None if invalid / empty.
    """
    if cfg is None:
        cfg = TemporalTeamConfig()

    if frame_bgr is None or frame_bgr.size == 0:
        return None

    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(w_img - 1), x2), min(float(h_img - 1), y2)
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None

    bw, bh = x2 - x1, y2 - y1
    pad = float(cfg.torso_x_pad) * bw
    cx0 = int(round(x1 + pad))
    cx1 = int(round(x2 - pad))
    ty0 = y1 + float(cfg.jersey_y0) * bh
    ty1 = y1 + float(cfg.jersey_y1) * bh
    ry0 = int(max(y1, min(ty0, y2 - 1.0)))
    ry1 = int(max(ry0 + 1.0, min(ty1, y2)))

    cx0 = max(0, min(cx0, w_img - 2))
    cx1 = max(cx0 + 1, min(cx1, w_img))
    ry0 = max(0, min(ry0, h_img - 2))
    ry1 = max(ry1 + 1, min(ry1, h_img))

    crop = frame_bgr[ry0:ry1, cx0:cx1]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return None
    return crop


def _normalize_hsv_rows(hsv: np.ndarray) -> np.ndarray:
    """Map OpenCV HSV to comparable scales in [0, 1] for distance-based clustering."""
    hsv = np.asarray(hsv, dtype=np.float64)
    out = np.empty_like(hsv)
    out[:, 0] = hsv[:, 0] / 180.0  # H in OpenCV: 0..179
    out[:, 1] = hsv[:, 1] / 255.0
    out[:, 2] = hsv[:, 2] / 255.0
    return out


def get_dominant_color(
    jersey_crop_bgr: np.ndarray,
    cfg: TemporalTeamConfig | None = None,
) -> np.ndarray | None:
    """
    Compute dominant jersey color in HSV using KMeans (k=2) on crop pixels.

    The larger cluster (by pixel count) is treated as the jersey fabric vs. skin/shadow.

    Returns:
        (3,) float32 HSV vector (H 0..179, S,V 0..255 OpenCV convention), or None if too few pixels.
    """
    if cfg is None:
        cfg = TemporalTeamConfig()

    if jersey_crop_bgr is None or jersey_crop_bgr.size == 0:
        return None

    hsv = cv2.cvtColor(jersey_crop_bgr, cv2.COLOR_BGR2HSV)
    flat = hsv.reshape(-1, 3).astype(np.float32)
    n = flat.shape[0]
    if n < cfg.per_crop_min_pixels:
        return None

    if n > cfg.per_crop_max_pixels:
        rng = np.random.default_rng(cfg.frame_kmeans_random_state)
        idx = rng.choice(n, size=cfg.per_crop_max_pixels, replace=False)
        flat = flat[idx]

    # KMeans in normalized space for more stable distances under lighting shifts on V.
    z = _normalize_hsv_rows(flat)
    # Solid-color crops: two clusters are degenerate; use mean HSV to avoid sklearn warnings.
    if float(np.std(z, axis=0).sum()) < 1e-4:
        m = np.mean(flat, axis=0)
        return np.asarray([m[0], m[1], m[2]], dtype=np.float32)

    k = int(cfg.per_crop_kmeans_k)
    try:
        km = KMeans(
            n_clusters=k,
            n_init=10,
            max_iter=100,
            random_state=cfg.frame_kmeans_random_state,
        )
        labels = km.fit_predict(z)
    except Exception as e:
        logger.debug("per-crop KMeans failed: %s", e)
        return np.mean(flat, axis=0).astype(np.float32)

    counts = np.bincount(labels, minlength=k)
    dominant_k = int(np.argmax(counts))
    # Center in original HSV units
    center_z = km.cluster_centers_[dominant_k]
    h = float(np.clip(center_z[0] * 180.0, 0.0, 179.0))
    s = float(np.clip(center_z[1] * 255.0, 0.0, 255.0))
    v = float(np.clip(center_z[2] * 255.0, 0.0, 255.0))
    return np.array([h, s, v], dtype=np.float32)


def assign_teams(
    dominant_colors: list[np.ndarray | None],
    cfg: TemporalTeamConfig | None = None,
    ref_centers_norm: np.ndarray | None = None,
) -> tuple[list[int | None], np.ndarray | None, np.ndarray]:
    """
    Cluster players into two teams from a list of per-player dominant HSV colors.

    Invalid entries (None) get label None.

    Aligns sklearn's arbitrary cluster ids to stable team slots (0 = Team A, 1 = Team B)
    by matching frame centers to EMA reference centers (reduces A/B swaps frame-to-frame).

    Returns:
        labels: length N, values 0/1 for team cluster or None if missing color.
        frame_centers_norm: (2,3) raw sklearn centers (normalized), or None if skipped.
        updated_ref: (2,3) EMA reference centers for the next frame.
    """
    if cfg is None:
        cfg = TemporalTeamConfig()

    n = len(dominant_colors)
    labels: list[int | None] = [None] * n
    valid_idx: list[int] = []
    rows: list[np.ndarray] = []

    for i, c in enumerate(dominant_colors):
        if c is None:
            continue
        rows.append(np.asarray(c, dtype=np.float32).reshape(3))
        valid_idx.append(i)

    if len(rows) == 0:
        ref = ref_centers_norm
        if ref is None:
            ref = np.zeros((2, 3), dtype=np.float64)
        return labels, None, ref

    # Single unlocked player: assign to nearest EMA team slot in normalized HSV (needs prior refs).
    if len(rows) == 1:
        ref = ref_centers_norm
        if ref is None or not np.any(ref):
            labels[valid_idx[0]] = 0
            if ref is None:
                ref = np.zeros((2, 3), dtype=np.float64)
            return labels, None, ref
        z0 = _normalize_hsv_rows(np.stack(rows, axis=0))
        d0 = float(np.linalg.norm(z0[0] - ref[0]))
        d1 = float(np.linalg.norm(z0[0] - ref[1]))
        labels[valid_idx[0]] = 0 if d0 <= d1 else 1
        return labels, None, ref

    x = np.stack(rows, axis=0)
    z = _normalize_hsv_rows(x)

    km = KMeans(
        n_clusters=2,
        n_init=cfg.frame_kmeans_n_init,
        max_iter=cfg.frame_kmeans_max_iter,
        random_state=cfg.frame_kmeans_random_state,
    )
    try:
        sub_labels = km.fit_predict(z)
    except Exception as e:
        logger.warning("frame KMeans failed: %s", e)
        ref = ref_centers_norm if ref_centers_norm is not None else np.zeros((2, 3), dtype=np.float64)
        return labels, None, ref

    centers = km.cluster_centers_.astype(np.float64)  # normalized space, arbitrary 0/1 semantics

    if ref_centers_norm is None or not np.any(ref_centers_norm):
        # First frame: anchor reference to this clustering (order is arbitrary but stable thereafter).
        ref_centers_norm = centers.copy()
        aligned = sub_labels.astype(np.int32)
    else:
        d_direct = float(
            np.linalg.norm(centers[0] - ref_centers_norm[0])
            + np.linalg.norm(centers[1] - ref_centers_norm[1])
        )
        d_swap = float(
            np.linalg.norm(centers[0] - ref_centers_norm[1])
            + np.linalg.norm(centers[1] - ref_centers_norm[0])
        )
        if d_direct <= d_swap:
            aligned = sub_labels.astype(np.int32)
            centers_aligned = centers.copy()
        else:
            # Swap so sklearn cluster 0 ↔ team slot 1; flip labels to match.
            aligned = (1 - sub_labels).astype(np.int32)
            centers_aligned = np.stack([centers[1], centers[0]], axis=0)

        mom = float(cfg.ref_center_momentum)
        ref_centers_norm = (1.0 - mom) * ref_centers_norm + mom * centers_aligned

    for j, vi in enumerate(valid_idx):
        labels[vi] = int(aligned[j])

    return labels, centers, ref_centers_norm


def update_team_history(
    history: dict[int, list[int]],
    track_id: int,
    team_label_binary: int | None,
    max_length: int | None,
) -> None:
    """
    Append a frame's team cluster (0 = Team A, 1 = Team B) for ``track_id``.

    Only appends if ``team_label_binary`` is not None.
    If ``max_length`` is None, the list grows unbounded (e.g. calibration window).
    Otherwise trims to the last ``max_length`` entries.
    """
    if team_label_binary is None:
        return
    tid = int(track_id)
    if tid not in history:
        history[tid] = []
    history[tid].append(int(team_label_binary))
    if max_length is not None and len(history[tid]) > max_length:
        history[tid] = history[tid][-max_length:]


def majority_vote_team(history_entry: list[int]) -> str | None:
    """Return TEAM_A or TEAM_B from the most common label, or None if empty."""
    if not history_entry:
        return None
    ones = sum(1 for x in history_entry if x == 1)
    zeros = len(history_entry) - ones
    return TEAM_B if ones > zeros else TEAM_A if zeros > ones else (TEAM_A if history_entry[-1] == 0 else TEAM_B)


def draw_team_bboxes(
    frame_bgr: np.ndarray,
    xyxy_list: list[tuple[int, int, int, int]],
    track_ids: list[int],
    team_labels: list[str],
    *,
    color_team_a: tuple[int, int, int] = (255, 120, 0),
    color_team_b: tuple[int, int, int] = (60, 180, 255),
    thickness: int = 2,
    font_scale: float = 0.55,
    highlight_track_id: int | None = None,
    highlight_color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """
    Draw axis-aligned rectangles and overlay ``ID | Team A/B`` for each player.

    BGR colors default to distinct blue-ish / orange-ish for teams.
    If ``highlight_track_id`` is set, that player gets a thicker highlight border.
    """
    out = frame_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x1, y1, x2, y2), tid, team in zip(xyxy_list, track_ids, team_labels):
        col = color_team_a if team == TEAM_A else color_team_b
        thk = thickness + 2 if highlight_track_id is not None and int(tid) == int(highlight_track_id) else thickness
        ccol = highlight_color if thk > thickness else col
        cv2.rectangle(out, (x1, y1), (x2, y2), ccol, thk)
        if thk > thickness:
            cv2.rectangle(out, (x1, y1), (x2, y2), col, thickness)
        label = f"ID {int(tid)} | {team}"
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        ty = max(0, y1 - 8)
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 4, ty + 2), (0, 0, 0), -1)
        cv2.putText(out, label, (x1 + 2, ty - 2), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


@dataclass
class TemporalTeamClassifier:
    """
    Stateful classifier: call ``update_frame`` once per video frame with player boxes + track IDs.

    Maintains per-track histories and EMA cluster references for stable Team A/B semantics.
    """

    cfg: TemporalTeamConfig = field(default_factory=TemporalTeamConfig)
    _history: dict[int, list[int]] = field(default_factory=dict)
    _last_team: dict[int, str] = field(default_factory=dict)
    _locked_team: dict[int, str] = field(default_factory=dict)
    _ref_centers_norm: np.ndarray | None = field(default=None)
    _calibration_locked: bool = field(default=False)
    _final_track_teams: dict[str, str] | None = field(default=None)

    def reset(self) -> None:
        """New video / stream."""
        self._history.clear()
        self._last_team.clear()
        self._locked_team.clear()
        self._ref_centers_norm = None
        self._calibration_locked = False
        self._final_track_teams = None

    def _apply_calibration_majority_lock(self) -> None:
        """After the calibration window, assign each track from majority vote and lock."""
        if self._calibration_locked:
            return
        self._calibration_locked = True
        for tid, hist in list(self._history.items()):
            tid = int(tid)
            mv = majority_vote_team(hist)
            team_str = mv if mv is not None else TEAM_A
            self._locked_team[tid] = team_str
            self._last_team[tid] = team_str
        # Tracks with no successful frame votes still have a provisional label in _last_team.
        for tid, lab in list(self._last_team.items()):
            tid = int(tid)
            if tid not in self._locked_team:
                self._locked_team[tid] = lab
        self._history.clear()

    def finalize_teams(self) -> dict[str, str]:
        """
        Call once at end of video analysis. If the video ended before the calibration window
        finished, this applies majority lock from all votes collected so far.

        Returns the final ``track_id -> Team A/B`` mapping (also used for JSON export).
        """
        cfg = self.cfg
        if cfg.calibration_duration_sec > 0 and not self._calibration_locked:
            self._apply_calibration_majority_lock()
        # Post-calibration late tracks already in _locked_team via freeze path
        out: dict[str, str] = {}
        for k in sorted(set(self._last_team.keys()) | set(self._locked_team.keys())):
            out[str(k)] = self.get_team(int(k))
        self._final_track_teams = out
        return out

    def export_track_team_json(self) -> dict[str, str]:
        """Final ``track_id -> team`` after ``finalize_teams()``; otherwise current best estimate."""
        if self._final_track_teams is not None:
            return dict(self._final_track_teams)
        merged = {**self._last_team, **self._locked_team}
        return {str(k): v for k, v in sorted(merged.items())}

    def get_team(self, track_id: int) -> str:
        """Team label for a track; frozen assignment takes precedence."""
        tid = int(track_id)
        if tid in self._locked_team:
            return self._locked_team[tid]
        return self._last_team.get(tid, TEAM_A)

    def is_in_calibration_window(self, time_sec: float | None) -> bool:
        """True while accumulating votes before majority lock (first ``calibration_duration_sec`` seconds)."""
        c = self.cfg
        if c.calibration_duration_sec <= 0 or time_sec is None:
            return False
        if self._calibration_locked:
            return False
        return float(time_sec) < float(c.calibration_duration_sec)

    def update_frame(
        self,
        frame_bgr: np.ndarray,
        player_xyxy: np.ndarray,
        track_ids: np.ndarray,
        confidences: np.ndarray | None = None,
        *,
        time_sec: float | None = None,
    ) -> tuple[list[str], dict[int, str]]:
        """
        Run full pipeline for one frame.

        Args:
            frame_bgr: BGR image.
            player_xyxy: (N, 4) float/int boxes.
            track_ids: (N,) stable IDs from ByteTrack.
            confidences: optional (N,) detection scores; gating if provided.
            time_sec: elapsed time in the clip (seconds). Required for calibration mode
                (``calibration_duration_sec`` > 0). First frame is typically ``0.0``.

        Returns:
            per_player_teams: length-N list of Team A/B aligned with input rows.
            smoothed_map: track_id -> current team label for this frame.
        """
        cfg = self.cfg
        n = len(player_xyxy)
        if n == 0 or len(track_ids) != n:
            return [], dict(self._last_team)

        use_calibration = cfg.calibration_duration_sec > 0 and time_sec is not None

        # End of calibration window: majority-lock before processing this frame's detections.
        if use_calibration and not self._calibration_locked:
            if time_sec >= float(cfg.calibration_duration_sec):
                self._apply_calibration_majority_lock()

        confidences = confidences if confidences is not None else np.ones((n,), dtype=np.float32)

        dominant_colors: list[np.ndarray | None] = []

        for i in range(n):
            tid = int(track_ids[i])
            # Locked tracks: do not re-extract jersey; they do not participate in frame clustering.
            if tid in self._locked_team:
                dominant_colors.append(None)
                continue

            xy = player_xyxy[i]
            x1, y1, x2, y2 = float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            conf = float(confidences[i])

            if area < cfg.min_bbox_area_px or conf < cfg.min_confidence:
                dominant_colors.append(None)
                continue

            crop = extract_jersey_crop(frame_bgr, xy, cfg)
            if crop is None:
                dominant_colors.append(None)
                continue

            dom = get_dominant_color(crop, cfg)
            dominant_colors.append(dom)

        frame_labels, _, self._ref_centers_norm = assign_teams(
            dominant_colors,
            cfg,
            ref_centers_norm=self._ref_centers_norm,
        )

        if use_calibration and not self._calibration_locked:
            # Calibration: collect votes; display running majority (no lock yet).
            for i in range(n):
                tid = int(track_ids[i])
                fl = frame_labels[i]
                if fl is not None:
                    update_team_history(self._history, tid, fl, max_length=None)
            for i in range(n):
                tid = int(track_ids[i])
                mv = majority_vote_team(self._history.get(tid, []))
                if mv is not None:
                    self._last_team[tid] = mv
                elif tid not in self._last_team:
                    self._last_team[tid] = TEAM_A
        elif use_calibration and self._calibration_locked:
            # After calibration: new tracks only — lock on first successful label if configured.
            if cfg.freeze_team_after_assignment:
                for i in range(n):
                    tid = int(track_ids[i])
                    if tid in self._locked_team:
                        continue
                    fl = frame_labels[i]
                    if fl is not None:
                        team_str = TEAM_A if fl == 0 else TEAM_B
                        self._locked_team[tid] = team_str
                        self._last_team[tid] = team_str
                    elif tid not in self._last_team:
                        self._last_team[tid] = TEAM_A
            else:
                for i in range(n):
                    tid = int(track_ids[i])
                    fl = frame_labels[i]
                    if fl is not None:
                        update_team_history(self._history, tid, fl, cfg.history_length)
                for i in range(n):
                    tid = int(track_ids[i])
                    if tid in self._locked_team:
                        continue
                    hist = self._history.get(tid, [])
                    mv = majority_vote_team(hist)
                    if mv is not None:
                        self._last_team[tid] = mv
                    elif tid not in self._last_team:
                        fl = frame_labels[i]
                        self._last_team[tid] = (
                            TEAM_A if fl == 0 else TEAM_B if fl is not None else TEAM_A
                        )
        else:
            # No calibration (``calibration_duration_sec == 0`` or ``time_sec`` omitted).
            if cfg.freeze_team_after_assignment:
                for i in range(n):
                    tid = int(track_ids[i])
                    if tid in self._locked_team:
                        continue
                    fl = frame_labels[i]
                    if fl is not None:
                        team_str = TEAM_A if fl == 0 else TEAM_B
                        self._locked_team[tid] = team_str
                        self._last_team[tid] = team_str
                    elif tid not in self._last_team:
                        self._last_team[tid] = TEAM_A
            else:
                for i in range(n):
                    tid = int(track_ids[i])
                    fl = frame_labels[i]
                    if fl is not None:
                        update_team_history(self._history, tid, fl, cfg.history_length)

                for i in range(n):
                    tid = int(track_ids[i])
                    hist = self._history.get(tid, [])
                    mv = majority_vote_team(hist)
                    if mv is not None:
                        self._last_team[tid] = mv
                    elif tid in self._last_team:
                        pass
                    else:
                        fl = frame_labels[i]
                        if fl is not None:
                            self._last_team[tid] = TEAM_A if fl == 0 else TEAM_B
                        else:
                            self._last_team[tid] = TEAM_A

        per_player_teams: list[str] = []
        for i in range(n):
            tid = int(track_ids[i])
            per_player_teams.append(self.get_team(tid))

        out_map = {**self._last_team, **{k: v for k, v in self._locked_team.items()}}
        return per_player_teams, out_map
