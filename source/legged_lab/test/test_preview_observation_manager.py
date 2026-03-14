from __future__ import annotations

import torch

from _isaaclab_test_app import simulation_app

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from legged_lab.managers import PreviewObservationManager


class _FakeSim:
    def is_playing(self) -> bool:
        return True


class _FakeEnv:
    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device
        self.sim = _FakeSim()
        self.history_source = torch.zeros(num_envs, device=device)


def _history_obs(env) -> torch.Tensor:
    return env.history_source.clone().unsqueeze(-1)


def _plain_obs(env) -> torch.Tensor:
    return (env.history_source * 10.0).clone().unsqueeze(-1)


def _make_env(num_envs: int = 2, device: str = "cpu"):
    return _FakeEnv(num_envs=num_envs, device=device)


@configclass
class _ObservationCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        concatenate_terms = False
        history = ObservationTermCfg(func=_history_obs, history_length=3, flatten_history_dim=False)
        plain = ObservationTermCfg(func=_plain_obs)

    policy: ObservationGroupCfg = PolicyCfg()


def test_preview_history_includes_current_obs_without_mutating_buffer() -> None:
    env = _make_env()
    obs_manager = PreviewObservationManager(_ObservationCfg(), env)

    env.history_source = torch.tensor([1.0, 10.0], device=env.device)
    obs_manager.compute_group("policy", update_history=True)
    env.history_source = torch.tensor([2.0, 20.0], device=env.device)
    obs_manager.compute_group("policy", update_history=True)

    history_buffer_before = obs_manager._group_obs_term_history_buffer["policy"]["history"].buffer.clone()
    current_length_before = obs_manager._group_obs_term_history_buffer["policy"]["history"].current_length.clone()

    env.history_source = torch.tensor([3.0, 30.0], device=env.device)
    preview = obs_manager.preview_group("policy")

    expected_preview = torch.tensor(
        [
            [[1.0], [2.0], [3.0]],
            [[10.0], [20.0], [30.0]],
        ],
        device=env.device,
    )
    assert torch.equal(preview["history"], expected_preview)
    assert torch.equal(obs_manager._group_obs_term_history_buffer["policy"]["history"].buffer, history_buffer_before)
    assert torch.equal(obs_manager._group_obs_term_history_buffer["policy"]["history"].current_length, current_length_before)


def test_preview_non_history_term_matches_processed_observation() -> None:
    env = _make_env()
    obs_manager = PreviewObservationManager(_ObservationCfg(), env)

    env.history_source = torch.tensor([4.0, 5.0], device=env.device)

    preview = obs_manager.preview_group("policy")
    computed = obs_manager.compute_group("policy", update_history=False)

    assert torch.equal(preview["plain"], computed["plain"])
