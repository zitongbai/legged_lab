from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
import isaaclab.envs.mdp as mdp

from legged_lab.tasks.position_tracking.gait_reward_based.mdp.commands import *

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_pos(env: ManagerBasedRLEnv, env_ids: Sequence[int], threshold: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired position.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded position.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    terrain: TerrainImporter = env.scene.terrain
    command: TerrainBasedPoseCommand = env.command_manager.get_term("target_position")
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the distance to the target position
    distance = torch.norm(command.robot_pos_w[env_ids, :] - command.target_pos_w[env_ids, :], dim=1)
    
    # robots that walked close enough to target position go to harder terrains
    move_up = distance <= threshold
    
    # robots that walked less than half of their required distance go to simpler terrains
    initial_distance = torch.norm(terrain.env_origins[env_ids, :] + asset.data.default_root_state[env_ids, :3] - command.target_pos_w[env_ids, :], dim=1)
    move_down = distance > initial_distance / 2.0
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

def override_value(env: ManagerBasedRLEnv, env_ids, current_value, value, num_steps):
    if env.common_step_counter > num_steps:
        if isinstance(current_value, dict) and isinstance(value, dict):
            current_value.update(value)
            return current_value
        else:
            # if not dict, directly return the new value
            return value
    return mdp.modify_term_cfg.NO_CHANGE
