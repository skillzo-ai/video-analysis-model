"""
Two-team player classification via jersey color clustering.
"""

from .config import TeamClusteringConfig
from .pipeline import annotate_frame, classify_teams, classify_teams_no_draw
from .temporal_team_classification import (
    TemporalTeamClassifier,
    TemporalTeamConfig,
    assign_teams,
    draw_team_bboxes,
    extract_jersey_crop,
    get_dominant_color,
    update_team_history,
)

__all__ = [
    "TeamClusteringConfig",
    "classify_teams",
    "classify_teams_no_draw",
    "annotate_frame",
    "TemporalTeamClassifier",
    "TemporalTeamConfig",
    "assign_teams",
    "draw_team_bboxes",
    "extract_jersey_crop",
    "get_dominant_color",
    "update_team_history",
]

