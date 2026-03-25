import gymnasium as gym

from . import agents

gym.register(
    id="LeggedLab-Isaac--Deepmimic-G1-v0",
    entry_point="legged_lab.envs:ManagerBasedAnimationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_deepmimic_env_cfg:G1DeepMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DeepMimicPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac--Deepmimic-G1-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAnimationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_deepmimic_env_cfg:G1DeepMimicEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DeepMimicPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac--Deepmimic-G1-Debug-v0",
    entry_point="legged_lab.envs:ManagerBasedAnimationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_deepmimic_env_cfg:G1DeepMimicEnvCfg_DEBUG",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DeepMimicPPORunnerCfg",
    },
)
