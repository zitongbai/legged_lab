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
    """Directory containing motion data files.

    Only supports reading .pkl files from this directory.
    """

    motion_data_weights: dict[str, float] = MISSING
    """Weights for the motion data in this term."""
