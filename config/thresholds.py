"""Tunable thresholds for basketball event detectors (image pixels / frame units)."""

# Minimum separation between previous and new ball owner centers to count a pass.
PASS_DISTANCE_THRESHOLD: float = 40.0

# In image coords, upward motion is negative vy; require vy <= -threshold.
SHOT_UPWARD_VELOCITY_THRESHOLD: float = 2.0

# Minimum normalized dot product between velocity and (hoop_center - ball) for a shot cue.
SHOT_HOOP_DOT_MIN: float = 0.15

# Frames to suppress repeated events of the same type after one fires.
EVENT_COOLDOWN_FRAMES: int = 12
