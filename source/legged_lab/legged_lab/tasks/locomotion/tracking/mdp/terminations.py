from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .commands import MotionTrackingCommand
from .rewards import _get_body_indices

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionTrackingCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=-1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionTrackingCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    command: MotionTrackingCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)
    robot_projected_gravity_b = math_utils.quat_rotate_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)
    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionTrackingCommand = env.command_manager.get_term(command_name)
    body_indices = _get_body_indices(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indices] - command.robot_body_pos_w[:, body_indices], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionTrackingCommand = env.command_manager.get_term(command_name)
    body_indices = _get_body_indices(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indices, 2] - command.robot_body_pos_w[:, body_indices, 2])
    return torch.any(error > threshold, dim=-1)


def motion_finished(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionTrackingCommand = env.command_manager.get_term(command_name)
    return command.motion_times >= command.motion_durations
