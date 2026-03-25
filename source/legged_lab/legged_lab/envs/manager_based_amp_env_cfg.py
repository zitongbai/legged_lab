from isaaclab.utils import configclass

from .manager_based_animation_env_cfg import ManagerBasedAnimationEnvCfg


@configclass
class ManagerBasedAmpEnvCfg(ManagerBasedAnimationEnvCfg):
    """Configuration for a AMP environment with the manager-based workflow."""

    terminal_obs_groups: tuple[str, ...] = ("disc",)
    """Observation groups to preview before reset and export through ``extras["terminal_obs"]``."""
