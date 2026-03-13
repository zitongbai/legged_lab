"""Legacy AMP event helpers.

These RSI helpers are currently not wired into the active AMP reset path.
AMP reset uses ``deepmimic.mdp.reset_from_ref`` via ``amp.mdp`` re-exports.
Keep this module only as documented legacy code until it is either revived or removed.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING, Literal
import random

import carb
import omni.physics.tensors.impl.api as physx
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from legged_lab.envs import ManagerBasedAmpEnv


def ref_state_init_root(
    env: ManagerBasedAmpEnv, 
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    motion_dataset: str | None = None,
    pos_rsi: bool = True,
):
    """Legacy Reference State Initialization (RSI) for the root of the robot.

    This helper is currently unused by the active AMP task configuration.
    Sample from the motion loader and set the root position and orientation.
    Refer to the paper of Adversarial Motion Priors (AMP) for more details.

    Args:
        env (AmpEnv): The manager-based env.
        env_ids (torch.Tensor): The env IDs to reset.
        asset_cfg (SceneEntityCfg, optional): The asset configuration. Defaults to SceneEntityCfg("robot").
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    num_envs = env_ids.shape[0]
    dt = env.cfg.sim.dt * env.cfg.decimation

    motion_dataset = _resolve_motion_dataset(env, motion_dataset)
    motion_loader = env.motion_data_manager.get_term(motion_dataset)
    motion_ids = motion_loader.sample_motions(num_envs)
    motion_times = motion_loader.sample_times(motion_ids, truncate_time_end=dt)
    motion_state_dict = motion_loader.get_motion_state(motion_ids, motion_times)
    
    lift_a_little = 0.05
    # lift the root position a little bit to avoid collision with the ground
    motion_state_dict["root_pos_w"][:, 2] += lift_a_little
    
    if not pos_rsi:
        motion_state_dict["root_pos_w"][:, :2] = 0.0    # no offset in x and y
    ref_root_pos_w = motion_state_dict["root_pos_w"] + env.scene.env_origins[env_ids]
    ref_root_quat = motion_state_dict["root_quat"]
    ref_root_vel_w = motion_state_dict["root_vel_w"]
    ref_root_ang_vel_w = motion_state_dict["root_ang_vel_w"]
    
    asset.write_root_pose_to_sim(
        torch.cat([ref_root_pos_w, ref_root_quat], dim=-1),
        env_ids=env_ids,
    )
    asset.write_root_velocity_to_sim(
        torch.cat([ref_root_vel_w, ref_root_ang_vel_w], dim=-1),
        env_ids=env_ids,
    )
    

def ref_state_init_dof(
    env: ManagerBasedAmpEnv, 
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    motion_dataset: str | None = None,
):
    """Legacy Reference State Initialization (RSI) for the joints (DoF) of the robot.

    This helper is currently unused by the active AMP task configuration.
    Sample from the motion loader and set the joint positions and velocities.
    Refer to the paper of Adversarial Motion Priors (AMP) for more details.

    Args:
        env (AmpEnv): The manager-based env.
        env_ids (torch.Tensor): The env IDs to reset.
        asset_cfg (SceneEntityCfg, optional): The asset configuration. Defaults to SceneEntityCfg("robot").
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    num_envs = env_ids.shape[0]
    dt = env.cfg.sim.dt * env.cfg.decimation
    
    motion_dataset = _resolve_motion_dataset(env, motion_dataset)
    motion_loader = env.motion_data_manager.get_term(motion_dataset)
    motion_ids = motion_loader.sample_motions(num_envs)
    motion_times = motion_loader.sample_times(motion_ids, truncate_time_end=dt)
    motion_state_dict = motion_loader.get_motion_state(motion_ids, motion_times)

    joint_pos = motion_state_dict["dof_pos"]
    joint_vel = motion_state_dict["dof_vel"]
    
    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def _resolve_motion_dataset(env: ManagerBasedAmpEnv, motion_dataset: str | None) -> str:
    """Resolve the motion-data term used by RSI events."""

    active_terms = env.motion_data_manager.active_terms
    if motion_dataset is not None:
        if motion_dataset not in active_terms:
            raise ValueError(f"Motion dataset '{motion_dataset}' not found in the active terms.")
        return motion_dataset

    if not active_terms:
        raise ValueError("No active motion datasets are configured.")

    term_weights = [env.motion_data_manager._term_cfgs[term_name].weight for term_name in active_terms]
    return random.choices(active_terms, weights=term_weights, k=1)[0]
