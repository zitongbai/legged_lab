# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to specify the symmetry in the observation and action space for go2."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]

# Observation layout metadata derived from position_env_cfg.
_HEIGHT_SCAN_ROWS = 11
_HEIGHT_SCAN_COLS = 31
_HEIGHT_SCAN_SIZE = _HEIGHT_SCAN_ROWS * _HEIGHT_SCAN_COLS

_POLICY_BASE_ANG_VEL = slice(0, 3)
_POLICY_PROJECTED_GRAVITY = slice(3, 6)
_POLICY_POSITION_COMMANDS = slice(6, 10)
_POLICY_REMAINING_TIME = slice(10, 11)
_POLICY_JOINT_POS = slice(11, 23)
_POLICY_JOINT_VEL = slice(23, 35)
_POLICY_ACTIONS = slice(35, 47)
_POLICY_HEIGHT_SCAN = slice(47, 47 + _HEIGHT_SCAN_SIZE)

_CRITIC_BASE_LIN_VEL = slice(0, 3)
_CRITIC_BASE_ANG_VEL = slice(3, 6)
_CRITIC_PROJECTED_GRAVITY = slice(6, 9)
_CRITIC_POSITION_COMMANDS = slice(9, 13)
_CRITIC_REMAINING_TIME = slice(13, 14)
_CRITIC_JOINT_POS = slice(14, 26)
_CRITIC_JOINT_VEL = slice(26, 38)
_CRITIC_ACTIONS = slice(38, 50)
_CRITIC_HEIGHT_SCAN = slice(50, 50 + _HEIGHT_SCAN_SIZE)
@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    four symmetrical transformations: original, left-right, front-back, and diagonal. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = obs.repeat(2)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(env.unwrapped, obs["policy"])
        # # -- front-back
        # obs_aug["policy"][2 * batch_size : 3 * batch_size] = _transform_policy_obs_front_back(
        #     env.unwrapped, obs["policy"]
        # )
        # # -- diagonal
        # obs_aug["policy"][3 * batch_size :] = _transform_policy_obs_front_back(
        #     env.unwrapped, obs_aug["policy"][batch_size : 2 * batch_size]
        # )

        # critic observation group
        if "critic" in obs.keys():
            # -- original
            obs_aug["critic"][:batch_size] = obs["critic"][:]
            # -- left-right
            obs_aug["critic"][batch_size : 2 * batch_size] = _transform_critic_obs_left_right(
                env.unwrapped, obs["critic"]
            )
            # # -- front-back
            # obs_aug["critic"][2 * batch_size : 3 * batch_size] = _transform_critic_obs_front_back(
            #     env.unwrapped, obs["critic"]
            # )
            # # -- diagonal
            # obs_aug["critic"][3 * batch_size :] = _transform_critic_obs_front_back(
            #     env.unwrapped, obs_aug["critic"][batch_size : 2 * batch_size]
            # )
    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
        # # -- front-back
        # actions_aug[2 * batch_size : 3 * batch_size] = _transform_actions_front_back(actions)
        # # -- diagonal
        # actions_aug[3 * batch_size :] = _transform_actions_front_back(actions_aug[batch_size : 2 * batch_size])
    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the policy observation tensor."""
    obs = obs.clone()
    device = obs.device
    obs[:, _POLICY_BASE_ANG_VEL] *= torch.tensor([-1, 1, -1], device=device)
    obs[:, _POLICY_PROJECTED_GRAVITY] *= torch.tensor([1, -1, 1], device=device)
    obs[:, _POLICY_POSITION_COMMANDS] *= torch.tensor([1, -1, 1, -1], device=device)
    obs[:, _POLICY_JOINT_POS] = _switch_go2_joints_left_right(obs[:, _POLICY_JOINT_POS])
    obs[:, _POLICY_JOINT_VEL] = _switch_go2_joints_left_right(obs[:, _POLICY_JOINT_VEL])
    obs[:, _POLICY_ACTIONS] = _switch_go2_joints_left_right(obs[:, _POLICY_ACTIONS])

    if "height_scan" in env.observation_manager.active_terms.get("policy", {}):
        obs[:, _POLICY_HEIGHT_SCAN] = (
            obs[:, _POLICY_HEIGHT_SCAN]
            .view(-1, _HEIGHT_SCAN_ROWS, _HEIGHT_SCAN_COLS)
            .flip(dims=[1])
            .view(-1, _HEIGHT_SCAN_SIZE)
        )

    return obs


def _transform_policy_obs_front_back(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a front-back symmetry transformation to the policy observation tensor."""
    obs = obs.clone()
    device = obs.device
    obs[:, _POLICY_BASE_ANG_VEL] *= torch.tensor([1, -1, -1], device=device)
    obs[:, _POLICY_PROJECTED_GRAVITY] *= torch.tensor([-1, 1, 1], device=device)
    obs[:, _POLICY_POSITION_COMMANDS] *= torch.tensor([-1, 1, -1], device=device)
    obs[:, _POLICY_JOINT_POS] = _switch_go2_joints_front_back(obs[:, _POLICY_JOINT_POS])
    obs[:, _POLICY_JOINT_VEL] = _switch_go2_joints_front_back(obs[:, _POLICY_JOINT_VEL])
    obs[:, _POLICY_ACTIONS] = _switch_go2_joints_front_back(obs[:, _POLICY_ACTIONS])

    if "height_scan" in env.observation_manager.active_terms.get("policy", {}):
        obs[:, _POLICY_HEIGHT_SCAN] = (
            obs[:, _POLICY_HEIGHT_SCAN]
            .view(-1, _HEIGHT_SCAN_ROWS, _HEIGHT_SCAN_COLS)
            .flip(dims=[2])
            .view(-1, _HEIGHT_SCAN_SIZE)
        )

    return obs


def _transform_critic_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the critic observation tensor."""
    obs = obs.clone()
    device = obs.device
    obs[:, _CRITIC_BASE_LIN_VEL] *= torch.tensor([1, -1, 1], device=device)
    obs[:, _CRITIC_BASE_ANG_VEL] *= torch.tensor([-1, 1, -1], device=device)
    obs[:, _CRITIC_PROJECTED_GRAVITY] *= torch.tensor([1, -1, 1], device=device)
    obs[:, _CRITIC_POSITION_COMMANDS] *= torch.tensor([1, -1, 1, -1], device=device)
    obs[:, _CRITIC_JOINT_POS] = _switch_go2_joints_left_right(obs[:, _CRITIC_JOINT_POS])
    obs[:, _CRITIC_JOINT_VEL] = _switch_go2_joints_left_right(obs[:, _CRITIC_JOINT_VEL])
    obs[:, _CRITIC_ACTIONS] = _switch_go2_joints_left_right(obs[:, _CRITIC_ACTIONS])

    if "height_scan" in env.observation_manager.active_terms.get("critic", {}):
        obs[:, _CRITIC_HEIGHT_SCAN] = (
            obs[:, _CRITIC_HEIGHT_SCAN]
            .view(-1, _HEIGHT_SCAN_ROWS, _HEIGHT_SCAN_COLS)
            .flip(dims=[1])
            .view(-1, _HEIGHT_SCAN_SIZE)
        )

    return obs


def _transform_critic_obs_front_back(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a front-back symmetry transformation to the critic observation tensor."""
    obs = obs.clone()
    device = obs.device
    obs[:, _CRITIC_BASE_LIN_VEL] *= torch.tensor([-1, 1, 1], device=device)
    obs[:, _CRITIC_BASE_ANG_VEL] *= torch.tensor([1, -1, -1], device=device)
    obs[:, _CRITIC_PROJECTED_GRAVITY] *= torch.tensor([-1, 1, 1], device=device)
    obs[:, _CRITIC_POSITION_COMMANDS] *= torch.tensor([-1, 1, -1], device=device)
    obs[:, _CRITIC_JOINT_POS] = _switch_go2_joints_front_back(obs[:, _CRITIC_JOINT_POS])
    obs[:, _CRITIC_JOINT_VEL] = _switch_go2_joints_front_back(obs[:, _CRITIC_JOINT_VEL])
    obs[:, _CRITIC_ACTIONS] = _switch_go2_joints_front_back(obs[:, _CRITIC_ACTIONS])

    if "height_scan" in env.observation_manager.active_terms.get("critic", {}):
        obs[:, _CRITIC_HEIGHT_SCAN] = (
            obs[:, _CRITIC_HEIGHT_SCAN]
            .view(-1, _HEIGHT_SCAN_ROWS, _HEIGHT_SCAN_COLS)
            .flip(dims=[2])
            .view(-1, _HEIGHT_SCAN_SIZE)
        )

    return obs


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    go2 robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_go2_joints_left_right(actions[:])
    return actions


def _transform_actions_front_back(actions: torch.Tensor) -> torch.Tensor:
    """Applies a front-back symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the front-back axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    go2 robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with front-back symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_go2_joints_front_back(actions[:])
    return actions


"""
Helper functions for symmetry.

For go2, the joint ordering is as follows:
[
    'FL_hip_joint',   'FR_hip_joint',   'RL_hip_joint',   'RR_hip_joint',
    'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
    'FL_calf_joint',  'FR_calf_joint',  'RL_calf_joint',  'RR_calf_joint'
]

Correspondingly, the joint ordering for the go2 robot is:

* FL = left front --> [0, 4, 8]
* RL = left rear --> [2, 6, 10]
* FR = right front --> [1, 5, 9]
* RR = right rear --> [3, 7, 11]
"""


def _switch_go2_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    joint_data_switched = torch.zeros_like(joint_data)
    # left <-- right
    joint_data_switched[..., [0, 4, 8, 2, 6, 10]] = joint_data[..., [1, 5, 9, 3, 7, 11]]
    # right <-- left
    joint_data_switched[..., [1, 5, 9, 3, 7, 11]] = joint_data[..., [0, 4, 8, 2, 6, 10]]

    # Flip the sign of the hip joints
    joint_data_switched[..., [0, 1, 2, 3]] *= -1.0

    return joint_data_switched


def _switch_go2_joints_front_back(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a front-back symmetry transformation to the joint data tensor."""
    joint_data_switched = torch.zeros_like(joint_data)
    # front <-- hind
    joint_data_switched[..., [0, 4, 8, 2, 6, 10]] = joint_data[..., [1, 5, 9, 3, 7, 11]]
    # hind <-- front
    joint_data_switched[..., [1, 5, 9, 3, 7, 11]] = joint_data[..., [0, 4, 8, 2, 6, 10]]

    # Flip the sign of the HFE and KFE joints
    joint_data_switched[..., 4:] *= -1

    return joint_data_switched
