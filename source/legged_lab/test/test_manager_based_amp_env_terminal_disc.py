from __future__ import annotations

from types import SimpleNamespace

import torch

from _isaaclab_test_app import simulation_app

from legged_lab.envs.manager_based_amp_env import ManagerBasedAmpEnv


def test_load_managers_uses_preview_observation_manager(monkeypatch) -> None:
    import legged_lab.envs.manager_based_amp_env as amp_env_module

    init_order: list[str] = []

    class _Manager:
        def __init__(self, label: str):
            self.label = label

    def _factory(label: str):
        class _Factory(_Manager):
            def __init__(self, cfg, env):
                super().__init__(label)
                init_order.append(label)
                self.cfg = cfg
                self.env = env

        return _Factory

    preview_manager_cls = _factory("observation")

    monkeypatch.setattr(amp_env_module, "MotionDataManager", _factory("motion_data"))
    monkeypatch.setattr(amp_env_module, "AnimationManager", _factory("animation"))
    monkeypatch.setattr(amp_env_module, "CommandManager", _factory("command"))
    monkeypatch.setattr(amp_env_module, "RecorderManager", _factory("recorder"))
    monkeypatch.setattr(amp_env_module, "ActionManager", _factory("action"))
    monkeypatch.setattr(amp_env_module, "PreviewObservationManager", preview_manager_cls)
    monkeypatch.setattr(amp_env_module, "TerminationManager", _factory("termination"))
    monkeypatch.setattr(amp_env_module, "RewardManager", _factory("reward"))
    monkeypatch.setattr(amp_env_module, "CurriculumManager", _factory("curriculum"))

    env = object.__new__(ManagerBasedAmpEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        motion_data="motion_cfg",
        animation="animation_cfg",
        commands="command_cfg",
        recorders="recorder_cfg",
        actions="action_cfg",
        observations="observation_cfg",
        terminations="termination_cfg",
        rewards="reward_cfg",
        curriculum="curriculum_cfg",
    )
    env.event_manager = SimpleNamespace(available_modes=[])
    env._configure_gym_env_spaces = lambda: setattr(env, "_gym_spaces_configured", True)

    env.load_managers()

    assert isinstance(env.observation_manager, preview_manager_cls)
    assert init_order == [
        "motion_data",
        "animation",
        "command",
        "recorder",
        "action",
        "observation",
        "termination",
        "reward",
        "curriculum",
    ]
    assert env._gym_spaces_configured is True


def test_step_exports_terminal_obs_and_keeps_post_reset_disc_obs() -> None:
    class _ActionManager:
        def process_action(self, action):
            self.last_action = action

        def apply_action(self):
            return None

    class _RecorderManager:
        active_terms = []

        def record_pre_step(self):
            return None

        def record_post_physics_decimation_step(self):
            return None

        def record_post_step(self):
            return None

        def record_pre_reset(self, env_ids):
            self.last_pre_reset = env_ids.clone()

        def record_post_reset(self, env_ids):
            self.last_post_reset = env_ids.clone()

    class _TerminationManager:
        terminated = torch.tensor([True, False])
        time_outs = torch.tensor([False, False])

        def compute(self):
            return self.terminated

    class _RewardManager:
        def compute(self, dt):
            self.last_dt = dt
            return torch.tensor([1.0, 2.0])

    class _ObservationManager:
        def preview(self):
            return {
                "disc": torch.tensor([[11.0], [22.0]]),
                "policy": torch.tensor([[111.0], [222.0]]),
            }

        def compute(self, update_history: bool):
            assert update_history is True
            return {
                "disc": torch.tensor([[101.0], [202.0]]),
                "policy": torch.tensor([[1001.0], [2002.0]]),
            }

    class _AnimationManager:
        def update(self, dt):
            self.last_dt = dt

    class _CommandManager:
        def compute(self, dt):
            self.last_dt = dt

    class _EventManager:
        available_modes = []

        def apply(self, mode: str, dt: float):
            self.last_mode = mode
            self.last_dt = dt

    class _Scene:
        def write_data_to_sim(self):
            return None

        def update(self, dt):
            self.last_dt = dt

    class _Sim:
        device = "cpu"

        def has_gui(self):
            return False

        def has_rtx_sensors(self):
            return False

        def step(self, render: bool = False):
            self.last_render = render

        def forward(self):
            self.forward_called = True

        def render(self):
            self.render_called = True

    env = object.__new__(ManagerBasedAmpEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        decimation=2,
        sim=SimpleNamespace(render_interval=1, dt=0.01),
        rerender_on_reset=False,
    )
    env.action_manager = _ActionManager()
    env.recorder_manager = _RecorderManager()
    env.termination_manager = _TerminationManager()
    env.reward_manager = _RewardManager()
    env.observation_manager = _ObservationManager()
    env.animation_manager = _AnimationManager()
    env.command_manager = _CommandManager()
    env.event_manager = _EventManager()
    env.scene = _Scene()
    env.sim = _Sim()
    env.extras = {}
    env._sim_step_counter = 0
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.common_step_counter = 0
    env._reset_idx = lambda env_ids: setattr(env, "_last_reset_ids", env_ids.clone())

    obs, rewards, terminated, time_outs, extras = env.step(torch.tensor([[0.1], [0.2]]))

    assert torch.equal(obs["disc"], torch.tensor([[101.0], [202.0]]))
    assert torch.equal(obs["policy"], torch.tensor([[1001.0], [2002.0]]))
    assert torch.equal(extras["terminal_obs"]["disc"], torch.tensor([[11.0], [202.0]]))
    assert set(extras["terminal_obs"].keys()) == {"disc"}
    assert torch.equal(terminated, torch.tensor([True, False]))
    assert torch.equal(time_outs, torch.tensor([False, False]))
    assert torch.equal(rewards, torch.tensor([1.0, 2.0]))
    assert torch.equal(env._last_reset_ids, torch.tensor([0]))


def test_step_previews_only_configured_terminal_obs_groups() -> None:
    class _ActionManager:
        def process_action(self, action):
            self.last_action = action

        def apply_action(self):
            return None

    class _RecorderManager:
        active_terms = []

        def record_pre_step(self):
            return None

        def record_post_physics_decimation_step(self):
            return None

        def record_post_step(self):
            return None

        def record_pre_reset(self, env_ids):
            return None

        def record_post_reset(self, env_ids):
            return None

    class _TerminationManager:
        terminated = torch.tensor([True, False])
        time_outs = torch.tensor([False, False])

        def compute(self):
            return self.terminated

    class _RewardManager:
        def compute(self, dt):
            return torch.tensor([1.0, 2.0])

    class _ObservationManager:
        def __init__(self):
            self.previewed_groups = []

        def preview_group(self, group_name: str):
            self.previewed_groups.append(group_name)
            if group_name == "disc":
                return torch.tensor([[11.0], [22.0]])
            if group_name == "policy":
                return torch.tensor([[111.0], [222.0]])
            raise AssertionError(f"Unexpected preview group: {group_name}")

        def compute(self, update_history: bool):
            assert update_history is True
            return {
                "disc": torch.tensor([[101.0], [202.0]]),
                "policy": torch.tensor([[1001.0], [2002.0]]),
            }

    class _AnimationManager:
        def update(self, dt):
            return None

    class _CommandManager:
        def compute(self, dt):
            return None

    class _EventManager:
        available_modes = []

        def apply(self, mode: str, dt: float):
            return None

    class _Scene:
        def write_data_to_sim(self):
            return None

        def update(self, dt):
            return None

    class _Sim:
        device = "cpu"

        def has_gui(self):
            return False

        def has_rtx_sensors(self):
            return False

        def step(self, render: bool = False):
            return None

        def forward(self):
            return None

        def render(self):
            return None

    env = object.__new__(ManagerBasedAmpEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        decimation=2,
        sim=SimpleNamespace(render_interval=1, dt=0.01),
        rerender_on_reset=False,
        terminal_obs_groups=("disc",),
    )
    env.action_manager = _ActionManager()
    env.recorder_manager = _RecorderManager()
    env.termination_manager = _TerminationManager()
    env.reward_manager = _RewardManager()
    env.observation_manager = _ObservationManager()
    env.animation_manager = _AnimationManager()
    env.command_manager = _CommandManager()
    env.event_manager = _EventManager()
    env.scene = _Scene()
    env.sim = _Sim()
    env.extras = {}
    env._sim_step_counter = 0
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.common_step_counter = 0
    env._reset_idx = lambda env_ids: None

    _, _, _, _, extras = env.step(torch.tensor([[0.1], [0.2]]))

    assert env.observation_manager.previewed_groups == ["disc"]
    assert set(extras["terminal_obs"].keys()) == {"disc"}
    assert torch.equal(extras["terminal_obs"]["disc"], torch.tensor([[11.0], [202.0]]))


def test_step_does_not_silently_ignore_terminal_obs_preview_failures() -> None:
    class _ActionManager:
        def process_action(self, action):
            self.last_action = action

        def apply_action(self):
            return None

    class _RecorderManager:
        active_terms = []

        def record_pre_step(self):
            return None

        def record_post_physics_decimation_step(self):
            return None

        def record_post_step(self):
            return None

        def record_pre_reset(self, env_ids):
            return None

        def record_post_reset(self, env_ids):
            return None

    class _TerminationManager:
        terminated = torch.tensor([True, False])
        time_outs = torch.tensor([False, False])

        def compute(self):
            return self.terminated

    class _RewardManager:
        def compute(self, dt):
            return torch.tensor([1.0, 2.0])

    class _ObservationManager:
        def preview(self):
            raise RuntimeError("preview failed")

        def compute(self, update_history: bool):
            return {"disc": torch.tensor([[101.0], [202.0]])}

    class _AnimationManager:
        def update(self, dt):
            return None

    class _CommandManager:
        def compute(self, dt):
            return None

    class _EventManager:
        available_modes = []

        def apply(self, mode: str, dt: float):
            return None

    class _Scene:
        def write_data_to_sim(self):
            return None

        def update(self, dt):
            return None

    class _Sim:
        device = "cpu"

        def has_gui(self):
            return False

        def has_rtx_sensors(self):
            return False

        def step(self, render: bool = False):
            return None

        def forward(self):
            return None

        def render(self):
            return None

    env = object.__new__(ManagerBasedAmpEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        decimation=2,
        sim=SimpleNamespace(render_interval=1, dt=0.01),
        rerender_on_reset=False,
    )
    env.action_manager = _ActionManager()
    env.recorder_manager = _RecorderManager()
    env.termination_manager = _TerminationManager()
    env.reward_manager = _RewardManager()
    env.observation_manager = _ObservationManager()
    env.animation_manager = _AnimationManager()
    env.command_manager = _CommandManager()
    env.event_manager = _EventManager()
    env.scene = _Scene()
    env.sim = _Sim()
    env.extras = {}
    env._sim_step_counter = 0
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.common_step_counter = 0
    env._reset_idx = lambda env_ids: None

    try:
        env.step(torch.tensor([[0.1], [0.2]]))
    except RuntimeError as exc:
        assert str(exc) == "preview failed"
    else:
        raise AssertionError("Expected terminal observation preview failure to propagate")
