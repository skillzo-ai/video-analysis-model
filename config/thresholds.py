"""Tunable thresholds for basketball event detectors (image pixels / frame units)."""

# Minimum separation between previous and new ball owner centers to count a pass.
PASS_DISTANCE_THRESHOLD: float = 40.0

# --- Shot (image y increases downward) ---
# Velocity = per-frame delta of tracked ball center in VideoProcessor.

# "Strict" path: strong upward motion + aim at rim anchor.
SHOT_MIN_UPWARD_SPEED: float = 1.0
SHOT_MIN_COS_TO_RIM: float = 0.12
SHOT_MIN_SPEED: float = 1.2

# "Flexible" path: weaker upward + must sit below rim (typical release geometry).
SHOT_FLEX_MIN_UPWARD_SPEED: float = 0.35
SHOT_FLEX_MIN_SPEED: float = 0.85
SHOT_FLEX_MIN_COS_TO_RIM: float = 0.06
# Ball center must be at least this many px below rim anchor y (image: larger y = lower on screen).
SHOT_FLEX_MIN_DEPTH_BELOW_RIM_PX: float = 8.0

SHOT_RIM_TOP_FRACTION: float = 0.22
SHOT_CONFIRM_FRAMES: int = 2
SHOT_COOLDOWN_FRAMES: int = 18

# --- Make (ball through rim opening) ---
# Opening = top-centered slice of hoop bbox (not full box — reduces early triggers).
MAKE_OPENING_WIDTH_FRAC: float = 0.52
MAKE_OPENING_HEIGHT_FRAC: float = 0.36
MAKE_OPENING_TOP_PAD_FRAC: float = 0.02

# Require ball to move toward opening (distance rim_anchor decreasing) on the entry frame or prior.
MAKE_REQUIRE_APPROACH: bool = True
MAKE_APPROACH_MAX_DIST_DELTA_PX: float = 2.5

# Cooldown after a make (frames).
MAKE_COOLDOWN_FRAMES: int = 28

# Pass / generic event cooldown (pass detector, legacy).
EVENT_COOLDOWN_FRAMES: int = 18

# Legacy aliases
SHOT_UPWARD_VELOCITY_THRESHOLD: float = SHOT_MIN_UPWARD_SPEED
SHOT_HOOP_DOT_MIN: float = SHOT_MIN_COS_TO_RIM
