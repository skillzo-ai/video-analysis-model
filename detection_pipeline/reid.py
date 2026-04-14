"""Re-exports appearance Re-ID from the reusable tracking package."""

from tracking.appearance import PlayerReID, cosine_similarity

__all__ = ["PlayerReID", "cosine_similarity"]
