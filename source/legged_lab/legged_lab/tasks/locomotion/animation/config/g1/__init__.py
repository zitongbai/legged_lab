import gymnasium as gym

##
# Register Gym environments.
##

# Pure motion-replay task: the robot is kinematically posed from motion data
# every step, so there is no policy/checkpoint and no ``rsl_rl_cfg_entry_point``.
# Run it with the zero-action agent (scripts/zero_agent.py).
gym.register(
    id="LeggedLab-Isaac-Animation-G1-v0",
    entry_point="legged_lab.envs:ManagerBasedAnimationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_anim_env_cfg:G1AnimEnvCfg",
    },
)
