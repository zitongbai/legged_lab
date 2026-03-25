from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm


def root_rot_tan_norm(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    root_quat = robot.data.root_quat_w
    root_rotm = math_utils.matrix_from_quat(root_quat)

    # use the first and last column of the rotation matrix as the tangent and normal vectors
    tan_vec = root_rotm[:, :, 0]  # (N, 3)
    norm_vec = root_rotm[:, :, 2]  # (N, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (N, 6)

    return obs


def key_body_pos_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=MISSING, preserve_order=True),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    key_body_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]  # shape: (num_envs, M, 3)
    root_pos_w = robot.data.root_pos_w  # shape: (num_envs, 3).
    root_quat = robot.data.root_quat_w  # shape: (num_envs, 4), w, x, y, z order.

    num_key_bodies = key_body_pos_w.shape[1]
    num_envs = root_pos_w.shape[0]

    key_body_pos_b = math_utils.quat_apply_inverse(
        root_quat.unsqueeze(1).expand(-1, num_key_bodies, -1),
        key_body_pos_w - root_pos_w.unsqueeze(1).expand(-1, num_key_bodies, -1),
    )

    return key_body_pos_b.reshape(num_envs, -1)


def ref_root_pos_error(
    env: ManagerBasedAnimationEnv,
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    abs_height: bool = True,
) -> torch.Tensor:
    """Compute the difference between robot root position and reference motion root position.

    The function calculates: reference_root_pos - current_robot_root_pos

    Args:
        env: The animation environment.
        animation: Name of the animation term to use as reference.
        asset_cfg: Configuration for the robot asset.
        abs_height: If True, use absolute height from reference motion (returns 3D position).
                   If False, only return horizontal displacement (2D: x, y only).

    Returns:
        Flattened tensor with shape:
        - (num_envs, num_steps * 3) if abs_height=True
        - (num_envs, num_steps * 2) if abs_height=False

    Note:
        Positive values indicate the reference motion is ahead/above the robot.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    ref_root_pos_w = animation_term.get_root_pos_w()  # shape: (num_envs, num_steps, 3)
    root_pos_w = robot.data.root_pos_w - env.scene.env_origins  # shape: (num_envs, 3)

    num_envs = root_pos_w.shape[0]

    # Compute position difference: ref - current
    # Broadcasting handles the dimension expansion automatically
    pos_diff = ref_root_pos_w - root_pos_w.unsqueeze(1)  # shape: (num_envs, num_steps, 3)

    if abs_height:
        # Replace relative z with absolute reference height
        pos_diff[:, :, 2] = ref_root_pos_w[:, :, 2]
        return pos_diff.reshape(num_envs, -1)  # shape: (num_envs, num_steps * 3)
    else:
        # Only return horizontal displacement (x, y)
        return pos_diff[:, :, :2].reshape(num_envs, -1)  # shape: (num_envs, num_steps * 2)


def ref_root_rot_tan_norm(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    ref_root_quat = animation_term.get_root_quat()  # shape: (num_envs, num_steps, 4)
    ref_root_rotm = math_utils.matrix_from_quat(ref_root_quat)  # shape: (num_envs, num_steps, 3, 3)
    ref_root_tan_vec = ref_root_rotm[:, :, :, 0]  # shape: (num_envs, num_steps, 3)
    ref_root_norm_vec = ref_root_rotm[:, :, :, 2]  # shape: (num_envs, num_steps, 3)
    obs = torch.cat([ref_root_tan_vec, ref_root_norm_vec], dim=-1)  # shape: (num_envs, num_steps, 6)

    if flatten_steps_dim:
        return obs.reshape(env.num_envs, -1)
    else:
        return obs


def ref_root_ang_vel_b(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs

    ref_root_ang_vel_w = animation_term.get_root_ang_vel_w()  # shape: (num_envs, num_steps, 3)
    ref_root_quat = animation_term.get_root_quat()  # shape: (num_envs, num_steps, 4)
    ref_root_ang_vel = math_utils.quat_apply_inverse(ref_root_quat, ref_root_ang_vel_w)

    if flatten_steps_dim:
        return ref_root_ang_vel.reshape(num_envs, -1)
    else:
        return ref_root_ang_vel


def ref_joint_pos(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    ref_dof_pos = animation_term.get_dof_pos()  # shape: (num_envs, num_steps, num_dofs)

    if flatten_steps_dim:
        return ref_dof_pos.reshape(env.num_envs, -1)
    else:
        return ref_dof_pos


def ref_joint_vel(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    ref_dof_vel = animation_term.get_dof_vel()  # shape: (num_envs, num_steps, num_dofs)

    if flatten_steps_dim:
        return ref_dof_vel.reshape(env.num_envs, -1)
    else:
        return ref_dof_vel


def ref_key_body_pos_b(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    ref_key_body_pos_b = animation_term.get_key_body_pos_b()  # shape: (num_envs, num_steps, num_key_bodies, 3)

    if flatten_steps_dim:
        return ref_key_body_pos_b.reshape(env.num_envs, -1)
    else:
        num_envs = ref_key_body_pos_b.shape[0]
        num_steps = ref_key_body_pos_b.shape[1]
        return ref_key_body_pos_b.reshape(num_envs, num_steps, -1)
