from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm


def ref_track_quat_error_exp(
    env: ManagerBasedAnimationEnv,
    std: float,
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    root_quat = robot.data.root_quat_w  # (N, 4)
    ref_root_quat = animation_term.get_root_quat()[:, 0, :]  # (N, 4)

    err = math_utils.quat_error_magnitude(root_quat, ref_root_quat)  # (N,)
    err_sq = torch.square(err)
    return torch.exp(-err_sq / std**2)  # (N,)


def ref_track_root_pos_w_error_exp(
    env: ManagerBasedAnimationEnv,
    std: float,
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    root_pos = robot.data.root_pos_w - env.scene.env_origins  # (N, 3)
    ref_root_pos = animation_term.get_root_pos_w()[:, 0, :]  # (N, 3)

    pos_err = torch.sum(torch.square(root_pos - ref_root_pos), dim=-1)  # (N,)
    return torch.exp(-pos_err / std**2)  # (N,)


def ref_track_root_vel_w_error_exp(
    env: ManagerBasedAnimationEnv,
    std: float,
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    root_vel = robot.data.root_lin_vel_w  # (N, 3)
    ref_root_vel = animation_term.get_root_vel_w()[:, 0, :]  # (N, 3)

    vel_err = torch.sum(torch.square(root_vel - ref_root_vel), dim=-1)  # (N,)
    return torch.exp(-vel_err / std**2)  # (N,)


def ref_track_root_ang_vel_w_error_exp(
    env: ManagerBasedAnimationEnv,
    std: float,
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    root_ang_vel = robot.data.root_ang_vel_w  # (N, 3)
    ref_root_ang_vel = animation_term.get_root_ang_vel_w()[:, 0, :]  # (N, 3)

    ang_vel_err = torch.sum(torch.square(root_ang_vel - ref_root_ang_vel), dim=-1)  # (N,)
    return torch.exp(-ang_vel_err / std**2)  # (N,)


def ref_track_key_body_pos_b_error_exp(
    env: ManagerBasedAnimationEnv,
    std: float,
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=MISSING),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    key_body_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]  # shape: (num_envs, M, 3)
    root_pos_w = robot.data.root_pos_w  # shape: (num_envs, 3).
    root_quat = robot.data.root_quat_w  # shape: (num_envs, 4), w, x, y, z order.

    num_key_bodies = key_body_pos_w.shape[1]
    key_body_pos_b = math_utils.quat_apply_inverse(
        root_quat.unsqueeze(1).expand(-1, num_key_bodies, -1).contiguous(),
        key_body_pos_w - root_pos_w.unsqueeze(1).expand(-1, num_key_bodies, -1).contiguous(),
    )

    ref_key_body_pos_b = animation_term.get_key_body_pos_b()[:, 0, :, :]  # shape: (num_envs, M, 3)

    key_body_pos_b_err = torch.sum(torch.square(key_body_pos_b - ref_key_body_pos_b), dim=-1)  # shape: (num_envs, M)
    key_body_pos_b_err_sum = torch.sum(key_body_pos_b_err, dim=-1)  # shape: (num_envs,)
    return torch.exp(-key_body_pos_b_err_sum / std**2)  # (N,)


def ref_track_dof_pos_error_exp(
    env: ManagerBasedAnimationEnv,
    std: float,
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    dof_pos = robot.data.joint_pos  # (N, D)
    ref_dof_pos = animation_term.get_dof_pos()[:, 0, :]  # (N, D)

    dof_pos_err = torch.sum(torch.square(dof_pos - ref_dof_pos), dim=-1)  # (N,)
    return torch.exp(-dof_pos_err / std**2)  # (N,)


def ref_track_dof_vel_error_exp(
    env: ManagerBasedAnimationEnv,
    std: float,
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    dof_vel = robot.data.joint_vel  # (N, D)
    ref_dof_vel = animation_term.get_dof_vel()[:, 0, :]  # (N, D)

    dof_vel_err = torch.sum(torch.square(dof_vel - ref_dof_vel), dim=-1)  # (N,)
    return torch.exp(-dof_vel_err / std**2)  # (N,)
