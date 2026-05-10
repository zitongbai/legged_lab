import gymnasium as gym

from . import agents


gym.register(
    id="LeggedLab-Isaac-Tracking-G1-v0",
    entry_point="legged_lab.envs:ManagerBasedMotionDataEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1TrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1TrackingPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-Tracking-G1-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedMotionDataEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1TrackingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1TrackingPPORunnerCfg",
    },
)
