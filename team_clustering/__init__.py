"""
Two-team player classification via jersey color clustering.
"""

from .config import TeamClusteringConfig
from .pipeline import annotate_frame, classify_teams, classify_teams_no_draw

__all__ = ["TeamClusteringConfig", "classify_teams", "classify_teams_no_draw", "annotate_frame"]

