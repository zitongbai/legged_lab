from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass


@configclass
class ManagerBasedMotionDataEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for an RL environment that uses motion data without animation buffers."""

    motion_data: object = MISSING
    """Motion data configuration.

    Please refer to :class:`legged_lab.managers.MotionDataManager` for details.
    """
