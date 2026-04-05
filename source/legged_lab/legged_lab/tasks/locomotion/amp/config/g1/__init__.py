import gymnasium as gym

from legged_lab.envs import ManagerBasedAmpEnv

from . import agents

gym.register(
    id="LeggedLab-Isaac-AMP-G1-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RslRlOnPolicyRunnerAmpCfg",
    },
)
