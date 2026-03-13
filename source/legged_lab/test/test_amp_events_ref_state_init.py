from __future__ import annotations

from types import SimpleNamespace

import torch

from _isaaclab_test_app import simulation_app

from legged_lab.tasks.locomotion.amp.mdp import events


class _MotionLoader:
    def __init__(self, motion_state: dict[str, torch.Tensor], motion_ids: torch.Tensor) -> None:
        self._motion_state = motion_state
        self._motion_ids = motion_ids
        self.sample_times_calls: list[tuple[torch.Tensor, float | None, float | None]] = []

    def sample_motions(self, n: int) -> torch.Tensor:
        assert n == self._motion_ids.shape[0]
        return self._motion_ids.clone()

    def sample_times(
        self,
        motion_ids: torch.Tensor,
        truncate_time_start: float | None = None,
        truncate_time_end: float | None = None,
    ) -> torch.Tensor:
        self.sample_times_calls.append((motion_ids.clone(), truncate_time_start, truncate_time_end))
        return torch.zeros_like(motion_ids, dtype=torch.float32)

    def get_motion_state(self, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> dict[str, torch.Tensor]:
        assert torch.equal(motion_ids, self._motion_ids)
        assert torch.equal(motion_times, torch.zeros_like(motion_ids, dtype=torch.float32))
        return {name: value.clone() for name, value in self._motion_state.items()}


class _MotionDataManager:
    def __init__(self, term_name: str, loader: _MotionLoader, weight: float = 1.0) -> None:
        self._term_name = term_name
        self._loader = loader
        self._term_cfgs = {term_name: SimpleNamespace(weight=weight)}

    @property
    def active_terms(self) -> list[str]:
        return [self._term_name]

    def get_term(self, term_name: str) -> _MotionLoader:
        assert term_name == self._term_name
        return self._loader


class _RootAsset:
    def __init__(self) -> None:
        self.pose_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.velocity_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def write_root_pose_to_sim(self, pose: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.pose_calls.append((pose.clone(), env_ids.clone()))

    def write_root_velocity_to_sim(self, velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.velocity_calls.append((velocity.clone(), env_ids.clone()))


class _DofAsset:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            soft_joint_pos_limits=torch.tensor(
                [
                    [[-0.5, 0.5], [-0.25, 0.25]],
                    [[-0.5, 0.5], [-0.25, 0.25]],
                ],
                dtype=torch.float32,
            ),
            soft_joint_vel_limits=torch.tensor(
                [
                    [1.0, 2.0],
                    [1.0, 2.0],
                ],
                dtype=torch.float32,
            ),
        )
        self.joint_state_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def write_joint_state_to_sim(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.joint_state_calls.append((joint_pos.clone(), joint_vel.clone(), env_ids.clone()))


class _Scene:
    def __init__(self, robot, env_origins: torch.Tensor | None = None) -> None:
        self._entities = {"robot": robot}
        self.env_origins = env_origins

    def __getitem__(self, key: str):
        return self._entities[key]


def test_ref_state_init_root_accepts_explicit_motion_dataset_and_uses_truncate_time_end() -> None:
    env_ids = torch.tensor([0, 1], dtype=torch.long)
    asset = _RootAsset()
    motion_state = {
        "root_pos_w": torch.tensor([[1.0, 2.0, 0.3], [4.0, 5.0, 0.6]], dtype=torch.float32),
        "root_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "root_vel_w": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32),
        "root_ang_vel_w": torch.tensor([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]], dtype=torch.float32),
    }
    loader = _MotionLoader(motion_state=motion_state, motion_ids=torch.tensor([3, 4], dtype=torch.long))
    env = SimpleNamespace(
        cfg=SimpleNamespace(sim=SimpleNamespace(dt=0.01), decimation=4),
        motion_data_manager=_MotionDataManager("motion_dataset", loader),
        scene=_Scene(asset, env_origins=torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=torch.float32)),
    )

    events.ref_state_init_root(env, env_ids, motion_dataset="motion_dataset")

    sample_motion_ids, truncate_time_start, truncate_time_end = loader.sample_times_calls[0]
    assert torch.equal(sample_motion_ids, torch.tensor([3, 4]))
    assert truncate_time_start is None
    assert truncate_time_end == 0.04
    pose, pose_env_ids = asset.pose_calls[0]
    velocity, velocity_env_ids = asset.velocity_calls[0]
    assert torch.equal(pose_env_ids, env_ids)
    assert torch.equal(velocity_env_ids, env_ids)
    assert torch.allclose(
        pose[:, :3],
        torch.tensor([[11.0, 22.0, 30.35], [44.0, 55.0, 60.65]], dtype=torch.float32),
    )
    assert torch.allclose(pose[:, 3:], motion_state["root_quat"])
    assert torch.allclose(
        velocity,
        torch.tensor(
            [[0.1, 0.2, 0.3, 0.7, 0.8, 0.9], [0.4, 0.5, 0.6, 1.0, 1.1, 1.2]],
            dtype=torch.float32,
        ),
    )


def test_ref_state_init_dof_uses_active_terms_without_get_term_weights_helper() -> None:
    env_ids = torch.tensor([0, 1], dtype=torch.long)
    asset = _DofAsset()
    motion_state = {
        "dof_pos": torch.tensor([[0.6, -0.4], [0.1, 0.3]], dtype=torch.float32),
        "dof_vel": torch.tensor([[1.5, -3.0], [0.2, 0.4]], dtype=torch.float32),
    }
    loader = _MotionLoader(motion_state=motion_state, motion_ids=torch.tensor([1, 0], dtype=torch.long))
    env = SimpleNamespace(
        cfg=SimpleNamespace(sim=SimpleNamespace(dt=0.005), decimation=2),
        motion_data_manager=_MotionDataManager("motion_dataset", loader),
        scene=_Scene(asset),
    )

    events.ref_state_init_dof(env, env_ids)

    sample_motion_ids, truncate_time_start, truncate_time_end = loader.sample_times_calls[0]
    assert torch.equal(sample_motion_ids, torch.tensor([1, 0]))
    assert truncate_time_start is None
    assert truncate_time_end == 0.01
    joint_pos, joint_vel, written_env_ids = asset.joint_state_calls[0]
    assert torch.equal(written_env_ids, env_ids)
    assert torch.allclose(joint_pos, torch.tensor([[0.5, -0.25], [0.1, 0.25]], dtype=torch.float32))
    assert torch.allclose(joint_vel, torch.tensor([[1.0, -2.0], [0.2, 0.4]], dtype=torch.float32))
