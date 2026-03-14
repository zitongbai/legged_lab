from __future__ import annotations

from collections.abc import Iterable

import torch

from isaaclab.managers import ObservationManager
from isaaclab.utils import noise
from isaaclab.utils.buffers import CircularBuffer


class PreviewObservationManager(ObservationManager):
    """Observation manager with a non-mutating preview API for group observations."""

    def preview(
        self, group_names: Iterable[str] | None = None
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if group_names is None:
            group_names = self._group_obs_term_names

        preview_buffer = {}
        for group_name in group_names:
            preview_buffer[group_name] = self.preview_group(group_name)
        return preview_buffer

    def preview_group(self, group_name: str) -> torch.Tensor | dict[str, torch.Tensor]:
        if group_name not in self._group_obs_term_names:
            raise ValueError(
                f"Unable to find the group '{group_name}' in the observation manager."
                f" Available groups are: {list(self._group_obs_term_names.keys())}"
            )

        group_term_names = self._group_obs_term_names[group_name]
        group_obs = dict.fromkeys(group_term_names, None)
        obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])

        for term_name, term_cfg in obs_terms:
            obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()

            if term_cfg.modifiers is not None:
                for modifier in term_cfg.modifiers:
                    obs = modifier.func(obs, **modifier.params)
            if isinstance(term_cfg.noise, noise.NoiseCfg):
                obs = term_cfg.noise.func(obs, term_cfg.noise)
            elif isinstance(term_cfg.noise, noise.NoiseModelCfg) and term_cfg.noise.func is not None:
                obs = term_cfg.noise.func(obs)
            if term_cfg.clip:
                obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
            if term_cfg.scale is not None:
                obs = obs.mul_(term_cfg.scale)

            if term_cfg.history_length > 0:
                circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]
                preview_buffer = CircularBuffer(
                    max_len=circular_buffer.max_length,
                    batch_size=circular_buffer.batch_size,
                    device=circular_buffer.device,
                )
                if circular_buffer._buffer is not None:
                    preview_buffer._buffer = circular_buffer._buffer.clone()
                    preview_buffer._pointer = circular_buffer._pointer
                    preview_buffer._num_pushes = circular_buffer._num_pushes.clone()
                preview_buffer.append(obs)
                if term_cfg.flatten_history_dim:
                    group_obs[term_name] = preview_buffer.buffer.reshape(self._env.num_envs, -1)
                else:
                    group_obs[term_name] = preview_buffer.buffer
            else:
                group_obs[term_name] = obs

        if self._group_obs_concatenate[group_name]:
            return torch.cat(list(group_obs.values()), dim=self._group_obs_concatenate_dim[group_name])
        return group_obs
