from __future__ import annotations

import os
import numpy as np
import torch
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerBase, ManagerTermBase

from .motion_data_term_cfg import MotionDataTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from legged_lab.utils.math import quat_slerp


class MotionDataTerm(ManagerTermBase):
    cfg: MotionDataTermCfg
    _env: ManagerBasedEnv

    def __init__(self, cfg: MotionDataTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        assert os.path.exists(cfg.motion_data_dir), f"Motion data directory {cfg.motion_data_dir} does not exist."

        self._load_motion_data()

        self._key_body_indices = [self.body_names.index(n) for n in cfg.key_body_names]
        self.num_key_bodies = len(self._key_body_indices)

    def _load_motion_data(self):
        motion_files = [f for f in os.listdir(self.cfg.motion_data_dir) if f.endswith(".npz")]
        if not motion_files:
            raise ValueError(f"No motion data files with .npz extension found in {self.cfg.motion_data_dir}.")

        self.motion_weights_dict = self.cfg.motion_data_weights

        self.motion_durations = []
        self.motion_fps = []
        self.motion_dt = []
        self.motion_num_frames = []
        self.motion_weights = []

        self.root_pos_w = []
        self.root_quat = []
        self.root_vel_w = []
        self.root_ang_vel_w = []
        self.dof_pos = []
        self.dof_vel = []
        self.body_pos_w = []
        self.body_quat_w = []
        self.body_lin_vel_w = []
        self.body_ang_vel_w = []

        self.body_names: list[str] = None

        for motion_name, motion_weight in self.motion_weights_dict.items():
            motion_file = f"{motion_name}.npz"
            if motion_file not in motion_files:
                raise ValueError(
                    f"Motion name {motion_name} defined in motion weights not found in motion data directory"
                    f" {self.cfg.motion_data_dir}. Available files: {motion_files}"
                )

            motion_path = os.path.join(self.cfg.motion_data_dir, motion_file)
            print(f"[Motion Data Manager] Loading motion data from {motion_path}...")
            data = np.load(motion_path, allow_pickle=True)
            if not isinstance(data, np.lib.npyio.NpzFile):
                raise ValueError(f"Motion data file {motion_file} is not a valid .npz file.")

            fps = int(data["fps"])
            dt = 1.0 / fps
            num_frames = data["root_pos"].shape[0]
            if num_frames < 2:
                raise ValueError(f"[MotionLoader] Motion has only {num_frames} frames.")
            duration = dt * (num_frames - 1)

            # Validate body_names consistency across files
            file_body_names = data["body_names"].tolist()
            if self.body_names is None:
                self.body_names = file_body_names
            elif self.body_names != file_body_names:
                raise ValueError(
                    f"body_names mismatch in {motion_file}. "
                    f"Expected {self.body_names}, got {file_body_names}."
                )

            self.motion_durations.append(duration)
            self.motion_fps.append(fps)
            self.motion_dt.append(dt)
            self.motion_num_frames.append(num_frames)
            self.motion_weights.append(motion_weight)

            def _to_tensor(key):
                return torch.from_numpy(data[key].astype(np.float32)).to(self.device)

            root_pos_w = _to_tensor("root_pos")
            root_quat = _to_tensor("root_rot")
            dof_pos = _to_tensor("dof_pos")
            body_pos_w = _to_tensor("body_pos_w")
            body_quat_w = _to_tensor("body_quat_w")
            body_lin_vel_w = _to_tensor("body_lin_vel_w")
            body_ang_vel_w = _to_tensor("body_ang_vel_w")

            # Derive root velocities from body index 0 (consistent with body data source)
            root_vel_w = body_lin_vel_w[:, 0, :]
            root_ang_vel_w = body_ang_vel_w[:, 0, :]

            # dof_vel: central finite differences over dof_pos
            dof_vel = torch.zeros_like(dof_pos)
            if num_frames >= 3:
                dof_vel[1:-1] = (dof_pos[2:] - dof_pos[:-2]) / (2.0 * dt)
                dof_vel[0] = dof_vel[1]
                dof_vel[-1] = dof_vel[-2]
            else:
                dof_vel[:-1] = (dof_pos[1:] - dof_pos[:-1]) / dt
                dof_vel[-1] = dof_vel[-2]

            self.root_pos_w.append(root_pos_w)
            self.root_quat.append(root_quat)
            self.root_vel_w.append(root_vel_w)
            self.root_ang_vel_w.append(root_ang_vel_w)
            self.dof_pos.append(dof_pos)
            self.dof_vel.append(dof_vel)
            self.body_pos_w.append(body_pos_w)
            self.body_quat_w.append(body_quat_w)
            self.body_lin_vel_w.append(body_lin_vel_w)
            self.body_ang_vel_w.append(body_ang_vel_w)

        self.motion_fps = torch.tensor(self.motion_fps, dtype=torch.float32, device=self.device)
        self.motion_dt = torch.tensor(self.motion_dt, dtype=torch.float32, device=self.device)
        self.motion_durations = torch.tensor(self.motion_durations, dtype=torch.float32, device=self.device)
        self.motion_num_frames = torch.tensor(self.motion_num_frames, dtype=torch.int32, device=self.device)
        self.motion_weights = torch.tensor(self.motion_weights, dtype=torch.float32, device=self.device)
        self.motion_weights = self.motion_weights / torch.sum(self.motion_weights)

        self.num_dofs = self.dof_pos[0].shape[1]
        self.num_bodies = len(self.body_names)

        self.root_pos_w = torch.cat(self.root_pos_w, dim=0)
        self.root_quat = torch.cat(self.root_quat, dim=0)
        self.root_vel_w = torch.cat(self.root_vel_w, dim=0)
        self.root_ang_vel_w = torch.cat(self.root_ang_vel_w, dim=0)
        self.dof_pos = torch.cat(self.dof_pos, dim=0)
        self.dof_vel = torch.cat(self.dof_vel, dim=0)
        self.body_pos_w = torch.cat(self.body_pos_w, dim=0)
        self.body_quat_w = torch.cat(self.body_quat_w, dim=0)
        self.body_lin_vel_w = torch.cat(self.body_lin_vel_w, dim=0)
        self.body_ang_vel_w = torch.cat(self.body_ang_vel_w, dim=0)

        num_motions = self.get_num_motions()
        self.motion_ids = torch.arange(num_motions, dtype=torch.long, device=self.device)

        lengths_shifted = self.motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self.motion_start_indices = torch.cumsum(lengths_shifted, dim=0)

    # Helper functions

    def get_num_motions(self) -> int:
        return self.motion_num_frames.shape[0]

    def get_total_duration(self) -> float:
        return torch.sum(self.motion_durations).item()

    def get_motion_durations(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self.motion_durations[motion_ids]

    def get_body_indices(self, body_names: list[str]) -> list[int]:
        """Return indices of the given body names in the stored body data."""
        return [self.body_names.index(n) for n in body_names]

    def sample_motions(self, n: int) -> torch.Tensor:
        return torch.multinomial(self.motion_weights, num_samples=n, replacement=True)

    def sample_times(
        self, motion_ids: torch.Tensor, truncate_time_start: float = None, truncate_time_end: float = None
    ) -> torch.Tensor:
        motion_durations = self.motion_durations[motion_ids]

        time_start = torch.zeros_like(motion_durations)
        time_end = motion_durations.clone()

        if truncate_time_start is not None:
            assert truncate_time_start >= 0
            time_start = torch.clamp(time_start + truncate_time_start, min=0.0, max=motion_durations)

        if truncate_time_end is not None:
            assert truncate_time_end >= 0
            time_end = torch.clamp(time_end - truncate_time_end, min=0.0)

        valid_range = time_end - time_start
        if torch.any(valid_range <= 0.0):
            print("[Warning] Some motions have invalid time range after truncation (start >= end).")
            valid_range = torch.clamp(valid_range, min=1e-6)

        phase = torch.rand(motion_ids.shape, device=self.device)
        return time_start + phase * valid_range

    def _calc_frame_blend(self, motion_ids: torch.Tensor, times: torch.Tensor):
        num_frames = self.motion_num_frames[motion_ids]
        motion_start_indices = self.motion_start_indices[motion_ids]
        motion_durations = self.motion_durations[motion_ids]

        phase = torch.clamp(times / motion_durations, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1).float()).long()
        frame_idx1 = torch.minimum(frame_idx0 + 1, num_frames - 1)
        blend = phase * (num_frames - 1).float() - frame_idx0.float()

        frame_idx0 = frame_idx0 + motion_start_indices
        frame_idx1 = frame_idx1 + motion_start_indices

        return frame_idx0, frame_idx1, blend

    def get_motion_state(self, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> dict[str, torch.Tensor]:
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)

        root_quat_0 = self.root_quat[frame_idx0]
        root_quat_1 = self.root_quat[frame_idx1]
        root_quat = quat_slerp(q0=root_quat_0, q1=root_quat_1, blend=blend)

        blend_1d = blend.unsqueeze(-1)
        root_pos_w = torch.lerp(self.root_pos_w[frame_idx0], self.root_pos_w[frame_idx1], blend_1d)
        root_vel_w = torch.lerp(self.root_vel_w[frame_idx0], self.root_vel_w[frame_idx1], blend_1d)
        root_vel_b = math_utils.quat_apply_inverse(root_quat, root_vel_w)
        root_ang_vel_w = torch.lerp(self.root_ang_vel_w[frame_idx0], self.root_ang_vel_w[frame_idx1], blend_1d)
        root_ang_vel_b = math_utils.quat_apply_inverse(root_quat, root_ang_vel_w)
        dof_pos = torch.lerp(self.dof_pos[frame_idx0], self.dof_pos[frame_idx1], blend_1d)
        dof_vel = torch.lerp(self.dof_vel[frame_idx0], self.dof_vel[frame_idx1], blend_1d)

        # key_body positions: slice from full body data using pre-built indices
        blend_3d = blend_1d.unsqueeze(-1)  # (N, 1, 1) for broadcasting over (N, K, 3)
        key_body_pos_w = torch.lerp(
            self.body_pos_w[frame_idx0][:, self._key_body_indices, :],
            self.body_pos_w[frame_idx1][:, self._key_body_indices, :],
            blend_3d,
        )
        key_body_pos_b = math_utils.quat_apply_inverse(
            root_quat.unsqueeze(1).expand(-1, self.num_key_bodies, -1),
            key_body_pos_w - root_pos_w.unsqueeze(1),
        )

        return {
            "root_pos_w": root_pos_w,
            "root_quat": root_quat,
            "root_vel_w": root_vel_w,
            "root_vel_b": root_vel_b,
            "root_ang_vel_w": root_ang_vel_w,
            "root_ang_vel_b": root_ang_vel_b,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "key_body_pos_b": key_body_pos_b,
        }


class MotionDataManager(ManagerBase):
    """Manager for motion data terms."""

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        if cfg is None:
            raise ValueError("MotionDataManager requires a valid configuration object.")

        self._terms: dict[str, MotionDataTerm] = {}
        self._term_cfgs: dict[str, MotionDataTermCfg] = {}

        super().__init__(cfg, env)

    def __str__(self) -> str:
        msg = f"<MotionDataManager> contains {len(self._terms)} active terms.\n"
        table = PrettyTable()
        table.title = "Motion Data Manager Terms"
        table.field_names = ["Index", "Motion Dataset", "Total Duration"]
        table.align["Motion Dataset"] = "l"
        table.align["Total Duration"] = "r"
        for index, (term_name, term) in enumerate(self._terms.items()):
            table.add_row([index, term_name, term.get_total_duration()])
        msg += table.get_string()
        msg += "\n"
        return msg

    @property
    def active_terms(self) -> list[str]:
        return list(self._terms.keys())

    def get_term(self, term_name: str) -> MotionDataTerm:
        if term_name not in self._terms:
            raise KeyError(f"Motion data term '{term_name}' not found.")
        return self._terms[term_name]

    def _prepare_terms(self):
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, MotionDataTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type MotionDataTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            term = MotionDataTerm(term_cfg, self._env)
            self._terms[term_name] = term
            self._term_cfgs[term_name] = term_cfg
