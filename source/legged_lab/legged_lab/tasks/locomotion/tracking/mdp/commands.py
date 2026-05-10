from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from legged_lab.managers import MotionDataTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionTrackingCommand(CommandTerm):
    """Reference-motion command backed by legged_lab motion data."""

    cfg: MotionTrackingCommandCfg

    def __init__(self, cfg: MotionTrackingCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.motion_data: MotionDataTerm = env.motion_data_manager.get_term(cfg.motion_data_term)

        robot_body_indices, robot_body_names = self.robot.find_bodies(cfg.body_names, preserve_order=True)
        if robot_body_names != cfg.body_names:
            raise ValueError(f"Robot body names mismatch. Expected {cfg.body_names}, got {robot_body_names}.")
        self.robot_body_indices = torch.tensor(robot_body_indices, dtype=torch.long, device=self.device)
        self.motion_body_indices = torch.tensor(
            self.motion_data.get_body_indices(cfg.body_names), dtype=torch.long, device=self.device
        )
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)

        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.motion_durations = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self._reference_state: dict[str, torch.Tensor] = {}
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0
        self.body_lin_vel_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_ang_vel_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)

        self.bin_failed_count = torch.zeros(
            self.motion_data.get_num_motions(), cfg.adaptive_num_bins, dtype=torch.float32, device=self.device
        )
        self._current_bin_failed = torch.zeros_like(self.bin_failed_count)
        self.kernel = torch.tensor(
            [cfg.adaptive_lambda**i for i in range(cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        for name in (
            "error_anchor_pos",
            "error_anchor_rot",
            "error_anchor_lin_vel",
            "error_anchor_ang_vel",
            "error_body_pos",
            "error_body_rot",
            "error_body_lin_vel",
            "error_body_ang_vel",
            "error_joint_pos",
            "error_joint_vel",
            "sampling_entropy",
            "sampling_top1_prob",
            "sampling_top1_bin",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.motion_ids[env_ids] = self.motion_data.sample_motions(len(env_ids))
        self.motion_durations[env_ids] = self.motion_data.get_motion_durations(self.motion_ids[env_ids])
        self.motion_times[env_ids] = self.motion_data.sample_times(self.motion_ids[env_ids])
        self.time_left[env_ids] = self.time_left[env_ids].uniform_(*self.cfg.resampling_time_range)
        self._update_reference_state()

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.dof_pos, self.dof_vel], dim=1)

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._reference_state["root_pos_w"] + self._env.scene.env_origins

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self._reference_state["root_quat"]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self._reference_state["root_vel_w"]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self._reference_state["root_ang_vel_w"]

    @property
    def dof_pos(self) -> torch.Tensor:
        return self._reference_state["dof_pos"]

    @property
    def dof_vel(self) -> torch.Tensor:
        return self._reference_state["dof_vel"]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._reference_state["body_pos_w"] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._reference_state["body_quat_w"]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._reference_state["body_lin_vel_w"]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._reference_state["body_ang_vel_w"]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w[:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_body_indices]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_body_indices]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_body_indices]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_body_indices]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _sample_motion_times_adaptive(self, env_ids: torch.Tensor, motion_ids: torch.Tensor) -> torch.Tensor:
        sampled_times = torch.zeros(len(motion_ids), dtype=torch.float32, device=self.device)
        for motion_id in torch.unique(motion_ids):
            env_mask = motion_ids == motion_id
            metric_env_ids = env_ids[env_mask]
            count = int(env_mask.sum().item())
            failed_count = self.bin_failed_count[motion_id]
            probabilities = failed_count + self.cfg.adaptive_uniform_ratio / float(self.cfg.adaptive_num_bins)
            probabilities = torch.nn.functional.pad(
                probabilities.unsqueeze(0).unsqueeze(0),
                (0, self.cfg.adaptive_kernel_size - 1),
                mode="replicate",
            )
            probabilities = torch.nn.functional.conv1d(probabilities, self.kernel.view(1, 1, -1)).view(-1)
            probabilities = probabilities / probabilities.sum()
            sampled_bins = torch.multinomial(probabilities, count, replacement=True)
            phase = (sampled_bins + torch.rand(count, device=self.device)) / self.cfg.adaptive_num_bins
            sampled_times[env_mask] = phase * self.motion_data.motion_durations[motion_id]

            entropy = -(probabilities * (probabilities + 1e-12).log()).sum() / math.log(self.cfg.adaptive_num_bins)
            pmax, imax = probabilities.max(dim=0)
            self.metrics["sampling_entropy"][metric_env_ids] = entropy
            self.metrics["sampling_top1_prob"][metric_env_ids] = pmax
            self.metrics["sampling_top1_bin"][metric_env_ids] = imax.float() / self.cfg.adaptive_num_bins
        return sampled_times

    def _record_failed_bins(self, env_ids: Sequence[int]) -> bool:
        if len(env_ids) == 0:
            return False
        if not hasattr(self._env, "termination_manager"):
            return False
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if not torch.any(episode_failed):
            return False
        failed_env_ids = env_ids[episode_failed]
        motion_ids = self.motion_ids[failed_env_ids]
        durations = torch.clamp(self.motion_durations[failed_env_ids], min=1e-6)
        bins = torch.clamp(
            (self.motion_times[failed_env_ids] / durations * self.cfg.adaptive_num_bins).long(),
            0,
            self.cfg.adaptive_num_bins - 1,
        )
        self._current_bin_failed.zero_()
        self._current_bin_failed.index_put_(
            (motion_ids, bins), torch.ones_like(bins, dtype=torch.float32), accumulate=True
        )
        return True

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        has_failures = False
        if self.cfg.adaptive_sampling:
            has_failures = self._record_failed_bins(env_ids)

        self.motion_ids[env_ids] = self.motion_data.sample_motions(len(env_ids))
        self.motion_durations[env_ids] = self.motion_data.get_motion_durations(self.motion_ids[env_ids])
        if self.cfg.adaptive_sampling:
            self.motion_times[env_ids] = self._sample_motion_times_adaptive(env_ids, self.motion_ids[env_ids])
        else:
            self.motion_times[env_ids] = self.motion_data.sample_times(self.motion_ids[env_ids])

        self._update_reference_state(env_ids)
        self._reset_robot_from_reference(env_ids)

        if self.cfg.adaptive_sampling and has_failures:
            self.bin_failed_count = (
                self.cfg.adaptive_alpha * self._current_bin_failed
                + (1.0 - self.cfg.adaptive_alpha) * self.bin_failed_count
            )
            self._current_bin_failed.zero_()

    def _get_selected_body_state(self, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.motion_data.get_motion_state(
            motion_ids,
            motion_times,
            body_indices=self.motion_body_indices,
            include_full_body=True,
        )

    def _update_reference_state(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        state = self._get_selected_body_state(self.motion_ids[env_ids], self.motion_times[env_ids])
        if not self._reference_state:
            self._reference_state = {
                key: torch.zeros((self.num_envs, *value.shape[1:]), dtype=value.dtype, device=self.device)
                for key, value in state.items()
            }
        for key, value in state.items():
            self._reference_state[key][env_ids] = value

        anchor_pos_w = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w.clone()
        delta_pos_w[..., 2] = anchor_pos_w[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(anchor_quat_w)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w)
        self.body_lin_vel_relative_w = quat_apply(delta_ori_w, self.body_lin_vel_w)
        self.body_ang_vel_relative_w = quat_apply(delta_ori_w, self.body_ang_vel_w)

    def _reset_robot_from_reference(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        root_pos = self.root_pos_w.clone()
        root_quat = self.root_quat_w.clone()
        root_lin_vel = self.root_lin_vel_w.clone()
        root_ang_vel = self.root_ang_vel_w.clone()

        pose_ranges = torch.tensor(
            [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
            device=self.device,
        )
        pose_noise = sample_uniform(pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += pose_noise[:, :3]
        root_quat[env_ids] = quat_mul(
            quat_from_euler_xyz(pose_noise[:, 3], pose_noise[:, 4], pose_noise[:, 5]), root_quat[env_ids]
        )

        velocity_ranges = torch.tensor(
            [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
            device=self.device,
        )
        velocity_noise = sample_uniform(
            velocity_ranges[:, 0], velocity_ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_lin_vel[env_ids] += velocity_noise[:, :3]
        root_ang_vel[env_ids] += velocity_noise[:, 3:]

        joint_pos = self.dof_pos.clone()
        joint_vel = self.dof_vel.clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, self.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )

        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_quat[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(-1)
        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_relative_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_relative_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(-1)
        self.metrics["error_joint_pos"] = torch.norm(self.dof_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.dof_vel - self.robot_joint_vel, dim=-1)

    def _update_command(self):
        self.motion_times += self._env.step_dt
        self._update_reference_state()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )
            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)


@configclass
class MotionTrackingCommandCfg(CommandTermCfg):
    """Configuration for motion tracking commands."""

    class_type: type = MotionTrackingCommand

    asset_name: str = MISSING
    motion_data_term: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.1, 0.1)

    adaptive_sampling: bool = True
    adaptive_num_bins: int = 64
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
