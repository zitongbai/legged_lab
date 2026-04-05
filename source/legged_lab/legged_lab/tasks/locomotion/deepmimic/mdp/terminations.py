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


def deviation_root_pos_w(
    env: ManagerBasedAnimationEnv,
    threshold: float,
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    root_pos_w = robot.data.root_pos_w - env.scene.env_origins  # (num_envs, 3)
    ref_root_pos_w = animation_term.get_root_pos_w()[:, 0, :]  # (num_envs, 3)

    dist = torch.norm(root_pos_w - ref_root_pos_w, dim=-1)  # (num_envs,)
    return dist > threshold


def deviation_key_body_pos_b(
    env: ManagerBasedAnimationEnv,
    threshold: float,
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=MISSING, preserve_order=True),
):
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    key_body_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]  # shape: (num_envs, M, 3)
    root_pos_w = robot.data.root_pos_w  # shape: (num_envs, 3).
    root_quat = robot.data.root_quat_w  # shape: (num_envs, 4), w, x, y, z order.

    num_key_bodies = key_body_pos_w.shape[1]

    key_body_pos_b = math_utils.quat_apply_inverse(
        root_quat.unsqueeze(1).expand(-1, num_key_bodies, -1).contiguous(), key_body_pos_w - root_pos_w.unsqueeze(1)
    )

    ref_key_body_pos_b = animation_term.get_key_body_pos_b()[:, 0, :, :]  # shape: (num_envs, M, 3)

    dist = torch.norm(key_body_pos_b - ref_key_body_pos_b, dim=-1)  # shape: (num_envs, M)
    max_dist, _ = torch.max(dist, dim=-1)  # shape: (num_envs,)
    return max_dist > threshold


def deviation_key_body_pos_w(
    env: ManagerBasedAnimationEnv,
    threshold: float,
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=MISSING, preserve_order=True),
):
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    key_body_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :] - env.scene.env_origins.unsqueeze(
        1
    )  # shape: (num_envs, M, 3)

    ref_key_body_pos_b = animation_term.get_key_body_pos_b()[:, 0, :, :]  # shape: (num_envs, M, 3)
    num_key_bodies = ref_key_body_pos_b.shape[1]
    ref_root_pos_w = animation_term.get_root_pos_w()[:, 0, :]  # (num_envs, 3)
    ref_root_quat = animation_term.get_root_quat()[:, 0, :]  # (num_envs, 4)
    ref_key_body_pos_w = ref_root_pos_w.unsqueeze(1) + math_utils.quat_apply(
        ref_root_quat.unsqueeze(1).expand(-1, num_key_bodies, -1).contiguous(), ref_key_body_pos_b
    )

    dist = torch.norm(ref_key_body_pos_w - key_body_pos_w, dim=-1)  # shape: (num_envs, M)
    max_dist, _ = torch.max(dist, dim=-1)  # shape: (num_envs,)
    return max_dist > threshold
