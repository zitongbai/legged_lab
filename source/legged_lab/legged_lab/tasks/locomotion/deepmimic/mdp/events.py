from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm


def reset_from_ref(
    env: ManagerBasedAnimationEnv,
    env_ids: torch.Tensor,
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset: float = 0.1,
):
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    offset = torch.tensor([0.0, 0.0, height_offset], device=env.device, dtype=torch.float32).unsqueeze(0)  # (1, 3)
    position = animation_term.get_root_pos_w(env_ids)[:, 0, :] + env.scene.env_origins[env_ids, :] + offset
    orientation = animation_term.get_root_quat(env_ids)[:, 0, :]
    lin_vel = animation_term.get_root_vel_w(env_ids)[:, 0, :]
    ang_vel = animation_term.get_root_ang_vel_w(env_ids)[:, 0, :]

    pos = torch.cat([position, orientation], dim=-1)
    vel = torch.cat([lin_vel, ang_vel], dim=-1)

    robot.write_root_pose_to_sim(pos, env_ids=env_ids)
    robot.write_root_velocity_to_sim(vel, env_ids=env_ids)

    dof_pos = animation_term.get_dof_pos(env_ids)[:, 0, :]
    dof_vel = animation_term.get_dof_vel(env_ids)[:, 0, :]
    robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
