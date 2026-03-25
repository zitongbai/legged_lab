from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass


@configclass
class ManagerBasedAnimationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for an animation environment with the manager-based workflow."""

    motion_data: object = MISSING
    """Motion data configuration for the animation environment.

    Please refer to the :class:`legged_lab.managers.MotionDataManager` class for more details.
    """
    animation: object = MISSING
    """Animation configuration for the animation environment.

    Please refer to the :class:`legged_lab.managers.AnimationManager` class for more details.
    """
