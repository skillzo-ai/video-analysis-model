"""
Persistent player identity: ByteTrack ``tracker_id`` -> stable ``global_id`` using
visual embedding (body appearance) plus optional jersey color and normalized bbox height.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

from .appearance import PlayerReID, cosine_similarity
from team_clustering.color_extraction import extract_color
from team_clustering.config import TeamClusteringConfig

from .schemas import TrackingConfig


def _lab_distance(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a.astype(np.float64) - b.astype(np.float64))
    return float(d)


def _color_similarity_from_lab_mean(a: np.ndarray, b: np.ndarray, sigma: float = 35.0) -> float:
    """Map LAB distance to [0, 1], higher is more similar."""
    d = _lab_distance(a, b)
    return float(np.exp(-d / max(sigma, 1e-6)))


class PlayerIdentityBridge:
    """
    Maintains ``id_map``: current ByteTrack tracker_id -> global_id, and
    ``id_memory``: global_id -> list of recent embeddings (and optional color / height).
    """

    def __init__(
        self,
        config: TrackingConfig,
        *,
        reid: PlayerReID | None = None,
        team_cfg: TeamClusteringConfig | None = None,
    ):
        self.cfg = config
        self.reid = reid or PlayerReID()
        self.team_cfg = team_cfg or TeamClusteringConfig(debug=False, draw_text=False)
        self.id_map: dict[int, int] = {}
        self.id_memory: dict[int, list[np.ndarray]] = {}
        self.id_colors: dict[int, np.ndarray] = {}
        self.id_heights: dict[int, float] = {}

    def reset(self) -> None:
        self.id_map.clear()
        self.id_memory.clear()
        self.id_colors.clear()
        self.id_heights.clear()

    def _fuse_score(
        self,
        sim_emb: float,
        sim_color: float | None,
        sim_h: float | None,
    ) -> float:
        w_e = float(self.cfg.embedding_weight)
        w_c = float(self.cfg.jersey_color_weight) if self.cfg.use_jersey_color else 0.0
        w_h = float(self.cfg.height_weight) if self.cfg.use_height_cue else 0.0
        if not self.cfg.use_jersey_color:
            sim_color = None
        if not self.cfg.use_height_cue:
            sim_h = None
        s = w_e * sim_emb
        if sim_color is not None:
            s += w_c * sim_color
        if sim_h is not None:
            s += w_h * sim_h
        # Renormalize if some weights zeroed
        w_sum = w_e + (w_c if sim_color is not None else 0.0) + (w_h if sim_h is not None else 0.0)
        return float(s / w_sum) if w_sum > 0 else sim_emb

    def _match_global_id(
        self,
        embedding: np.ndarray | None,
        jersey_lab: np.ndarray | None,
        height_norm: float | None,
    ) -> int | None:
        best_gid: int | None = None
        best_score = -1.0
        thr = float(self.cfg.reid_similarity_threshold)

        for gid, embs in self.id_memory.items():
            if not embs:
                continue
            tail = embs[-int(self.cfg.embedding_compare_last) :]
            sims = [cosine_similarity(embedding, e) for e in tail] if embedding is not None else []
            sim_emb = max(sims) if sims else 0.0

            sim_color: float | None = None
            if self.cfg.use_jersey_color and jersey_lab is not None and gid in self.id_colors:
                sim_color = _color_similarity_from_lab_mean(jersey_lab, self.id_colors[gid])

            sim_h: float | None = None
            if self.cfg.use_height_cue and height_norm is not None and gid in self.id_heights:
                sim_h = float(1.0 - min(abs(height_norm - self.id_heights[gid]), 1.0))

            fused = self._fuse_score(sim_emb, sim_color, sim_h)
            if fused > best_score:
                best_score = fused
                best_gid = gid

        if best_gid is not None and best_score >= thr:
            return int(best_gid)
        return None

    def update(
        self,
        frame_bgr: np.ndarray,
        detections: sv.Detections,
    ) -> tuple[sv.Detections, list[dict[str, object] | None]]:
        """
        Remap ``tracker_id`` for player class to stable global ids.
        Returns updated detections and one entry per detection row (``None`` if not a player).
        """
        h_frame = float(frame_bgr.shape[0])
        cls = detections.class_id
        tid = detections.tracker_id
        if cls is None or tid is None:
            return detections, [None] * len(detections)

        new_ids: list[int] = []
        meta: list[dict[str, object] | None] = [None] * len(detections)

        for i in range(len(detections)):
            if int(cls[i]) != int(self.cfg.player_class_id):
                new_ids.append(int(tid[i]))
                continue

            bbox = detections.xyxy[i].astype(int)
            crop = frame_bgr[max(0, bbox[1]) : bbox[3], max(0, bbox[0]) : bbox[2]]
            embedding = self.reid.extract_embedding(crop)

            jersey_lab = None
            if self.cfg.use_jersey_color and crop.size > 0:
                jersey_lab = extract_color(crop, self.team_cfg)

            height_norm = None
            if self.cfg.use_height_cue and h_frame > 1:
                height_norm = float((bbox[3] - bbox[1]) / h_frame)

            tr = int(tid[i])

            cues: dict[str, object] = {
                "has_embedding": embedding is not None,
                "jersey_lab": jersey_lab.tolist() if jersey_lab is not None else None,
                "height_norm": height_norm,
            }

            if tr not in self.id_map:
                matched = None
                if embedding is not None or (self.cfg.use_jersey_color and jersey_lab is not None):
                    matched = self._match_global_id(embedding, jersey_lab, height_norm)
                if matched is not None:
                    self.id_map[tr] = matched
                else:
                    # New global identity: use this tracker id as canonical gid seed
                    self.id_map[tr] = tr

            gid = int(self.id_map[tr])

            if embedding is not None:
                if gid not in self.id_memory:
                    self.id_memory[gid] = []
                self.id_memory[gid].append(embedding)
                hist = self.id_memory[gid]
                cap = int(self.cfg.max_embedding_history)
                if len(hist) > cap:
                    self.id_memory[gid] = hist[-cap:]

            if self.cfg.use_jersey_color and jersey_lab is not None:
                self.id_colors[gid] = jersey_lab
            if self.cfg.use_height_cue and height_norm is not None:
                self.id_heights[gid] = height_norm

            cues["global_id"] = gid
            new_ids.append(gid)
            meta[i] = cues

        out = detections
        out.tracker_id = np.asarray(new_ids, dtype=np.int32)
        return out, meta
