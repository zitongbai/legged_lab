from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class MotionDataTermCfg:
    """
    Configuration for the motion data term in the motion data manager.
    """

    weight: float = 1.0
    """Weight of this term in the motion data manager."""

    motion_data_dir: str = MISSING
    """Directory containing motion data files (.npz format)."""

    motion_data_weights: dict[str, float] = MISSING
    """Weights for the motion data in this term."""

    key_body_names: list[str] = MISSING
    """Names of key bodies for AMP/DeepMimic observations.

    Must match body names stored in the motion data files.
    Order must align with the robot body_names used in downstream tasks.
    """
