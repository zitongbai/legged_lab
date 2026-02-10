from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import RayCaster, ContactSensor
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from legged_lab.tasks.position_tracking.gait_reward_based.mdp.commands import *

def task_reward(env: ManagerBasedRLEnv, command_name: str, Tr: float = 1.0) -> torch.Tensor:
    """Compute the task reward based on the distance to the target position and the remaining time.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        command_name (str): The name of the command to retrieve target positions.
        Tr (float): The time window before the end of the episode to start rewarding.

    Returns:
        torch.Tensor: The computed reward tensor of shape (num_envs,).
    """
    command: TerrainBasedPoseCommand = env.command_manager.get_term(command_name)
    robot_pos = command.robot_pos_w
    target_pos = command.target_pos_w

    distance = torch.norm(robot_pos - target_pos, dim=-1)  # (num_envs,)

    # Condition for when to apply the reward
    condition = env.episode_length_buf * env.step_dt >= env.max_episode_length_s - Tr
    
    # Calculate reward using torch.where for vectorized operation
    reward = torch.where(condition, 1.0 - 0.5 * distance, 0.0)
    
    return reward

def exploration_reward(env: ManagerBasedRLEnv, command_name: str, Tr: float = 1.0) -> torch.Tensor:
    """Compute the exploration reward based on the orientation of the robot.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        command_name (str): The name of the command to retrieve target positions.
        Tr (float): The time window before the end of the episode to start rewarding.

    Returns:
        torch.Tensor: The computed reward tensor of shape (num_envs,).
    """
    command: TerrainBasedPoseCommand = env.command_manager.get_term(command_name)
    robot_vel = command.robot_velocity_w # (num_envs, 3)
    target_vec = command.target_pos_w - command.robot_pos_w # (num_envs, 3)
    distance = torch.norm(target_vec, dim=-1)  # (num_envs,)

    # Condition for when to apply the reward
    condition = (env.episode_length_buf * env.step_dt >= env.max_episode_length_s - Tr) & (distance <= 1.0) & (torch.norm(robot_vel, dim=-1) < 0.2)
    
    # Calculate cosine similarity
    # Dot product for the numerator
    dot_product = torch.sum(robot_vel * target_vec, dim=-1)
    # Norms for the denominator
    robot_vel_norm = torch.norm(robot_vel, dim=-1)
    target_vec_norm = torch.norm(target_vec, dim=-1)
    
    # Calculate reward using torch.where for vectorized operation
    cosine_sim = dot_product / (robot_vel_norm * target_vec_norm + 1e-8)
    
    reward = torch.where(condition, 0.0, cosine_sim)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # scale with gravity projection
    return reward

def stalling_penalty(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Compute the stalling penalty based on the robot's velocity.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        command_name (str): The name of the command to retrieve target positions.

    Returns:
        torch.Tensor: The computed penalty tensor of shape (num_envs,).
    """
    command: TerrainBasedPoseCommand = env.command_manager.get_term(command_name)
    speed = torch.norm(command.robot_velocity_w, dim=-1)  # (num_envs,)
    distance = torch.norm(command.robot_pos_w - command.target_pos_w, dim=-1)  # (num_envs,)

    # Condition for when to apply the reward
    condition = (speed < 0.2) & (distance > 0.3)
    
    # Calculate reward using torch.where for vectorized operation
    reward = torch.where(condition, 1.0, 0.0)

    return reward

def feet_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty for high feet acceleration"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    feet_acc = asset.data.body_acc_w[:, asset_cfg.body_ids, :3]  # (num_envs, num_feet, 3)
    penalty = torch.norm(feet_acc, dim=-1)  # (num_envs, num_feet)
    reward = torch.sum(torch.square(penalty), dim=-1)  # (num_envs,)
    return reward

def base_height_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L1 norm.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]
    target_height = robot.data.default_root_state[0, 2]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_pos_w[:, 2]  # fallback to current height if sensor data is invalid
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L1 squared penalty
    reward = torch.abs(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    return reward

def base_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty for base acceleration"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    base_acc_lin = asset.data.body_acc_w[:, 0, :3]  # (num_envs, 3)
    base_acc_ang = asset.data.body_acc_w[:, 0, 3:]  # (num_envs, 3)
    reward = torch.square(torch.norm(base_acc_lin, dim=-1)) + 0.02 * torch.square(torch.norm(base_acc_ang, dim=-1))  # (num_envs,)
    return reward

def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str, Tr: float = 1.0) -> torch.Tensor:
    """Compute the absolute heading error between the robot's heading and the commanded heading.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        command_name (str): The name of the command to retrieve target headings.

    Returns:
        torch.Tensor: The computed absolute heading error tensor of shape (num_envs,).
    """
    # command: TerrainBasedPoseCommand = env.command_manager.get_term(command_name)
    # distance = torch.norm(command.robot_pos_w - command.target_pos_w, dim=-1)  # (num_envs,)

    # condition = (env.episode_length_buf * env.step_dt >= env.max_episode_length_s - Tr) & (distance <= 0.5)
    command: TerrainBasedPoseCommand = env.command_manager.get_term(command_name)
    heading_b = command.target_heading_b  # (num_envs,)
    reward = 1 - 0.5 * heading_b.abs()
    return reward

def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty for variance in feet air time"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]  # (num_envs, num_feet)
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]  # (num_envs, num_feet)
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1)  # (num_envs,)
    
def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward
    
def flat_orientation_xy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-component of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)  # (num_envs,)

    
class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    dis_threshold: float = 0.25,
    heading_threshold: float = 0.5,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    
    # Calculate height error: only penalize if foot is LOWER than target height
    # error = max(0, target_height - current_height)
    foot_z_target_error = torch.clamp(target_height - footpos_in_body_frame[:, :, 2], min=0.0) # (num_envs, num_feet)
    
    # Identify swing phase: foot has significant horizontal velocity
    is_swing = torch.norm(footvel_in_body_frame[:, :, :2], dim=2) > 0.1 # (num_envs, num_feet)

    # We sum over all feet
    reward = torch.sum(foot_z_target_error * is_swing, dim=1) # (num_envs,)
    
    command: TerrainBasedPoseCommand = env.command_manager.get_term(command_name)
    distance = torch.norm(command.robot_pos_w - command.target_pos_w, dim=-1)  # (num_envs,)
    heading_error = torch.abs(command.target_heading_b)  # (num_envs,)
    condition = (distance < dis_threshold) & (heading_error < heading_threshold)
    reward = torch.where(condition, 0.0, reward)
    
    # Scale with gravity projection (optional, but good for stability)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_air_time_1(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float,
    dis_threshold: float = 0.25, heading_threshold: float = 0.5
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    command: TerrainBasedPoseCommand = env.command_manager.get_term(command_name)
    distance = torch.norm(command.robot_pos_w - command.target_pos_w, dim=-1)  # (num_envs,)
    heading_error = torch.abs(command.target_heading_b)  # (num_envs,)
    condition = (distance < dis_threshold) & (heading_error < heading_threshold)
    reward = torch.where(condition, 0.0, reward)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_edge_penalty(
    env: ManagerBasedRLEnv,
    FL_ray_sensor_cfg: SceneEntityCfg,
    FR_ray_sensor_cfg: SceneEntityCfg,
    RL_ray_sensor_cfg: SceneEntityCfg,
    RR_ray_sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    edge_height_thresh: float = 0.05,
) -> torch.Tensor:
    """Penalize if the feet are close to the edge of the terrain.
    
    This is detected by checking the height variance/difference in the vicinity of the feet.
    """
    FL_ray_sensor: RayCaster = env.scene.sensors[FL_ray_sensor_cfg.name]
    FR_ray_sensor: RayCaster = env.scene.sensors[FR_ray_sensor_cfg.name]
    RL_ray_sensor: RayCaster = env.scene.sensors[RL_ray_sensor_cfg.name]
    RR_ray_sensor: RayCaster = env.scene.sensors[RR_ray_sensor_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
   
    # ray_hits shape: (num_envs， 4, num_rays, 3)
    FL_ray_hits = FL_ray_sensor.data.ray_hits_w.view(env.num_envs, 1, -1, 3)
    FR_ray_hits = FR_ray_sensor.data.ray_hits_w.view(env.num_envs, 1, -1, 3)
    RL_ray_hits = RL_ray_sensor.data.ray_hits_w.view(env.num_envs, 1, -1, 3)
    RR_ray_hits = RR_ray_sensor.data.ray_hits_w.view(env.num_envs, 1, -1, 3)
    ray_hits = torch.cat([FL_ray_hits, FR_ray_hits, RL_ray_hits, RR_ray_hits], dim=1)
    
    # Get heights
    scan_heights = ray_hits[..., 2] # (num_envs, 4, num_rays)
    
    # Calculate height difference (max - min) in the scan
    # We use a robust metric, e.g., std dev or range.
    height_range = torch.max(scan_heights, dim=-1)[0] - torch.min(scan_heights, dim=-1)[0] # (num_envs, 4)
    
    # Check if foot is in contact
    # We use the robot's contact force data
    contacts = contact_sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0 # (num_envs, num_feet)
    
    # Penalty
    # If in contact AND height_range > threshold
    is_edge = height_range > edge_height_thresh
    
    penalty = torch.sum(torch.where(contacts & is_edge, 1.0, 0.0), dim=-1)
    
    return penalty
