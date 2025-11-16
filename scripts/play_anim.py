import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument(
    "--robot", 
    type=str,
    default="g1_29dof",
    help="The robot name to be used.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.assets import Articulation

from legged_lab.envs import ManagerBasedAnimationEnv

##
# Pre-defined configs
##
if args_cli.robot == "g1_29dof":
    from legged_lab.tasks.locomotion.animation.config.g1.g1_anim_env_cfg import G1AnimEnvCfg as AnimEnvCfg
else:
    raise ValueError(f"Robot {args_cli.robot} not supported.")


def main():
    env_cfg = AnimEnvCfg()
    
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    env = ManagerBasedAnimationEnv(cfg=env_cfg)
    robot_anim: Articulation = env.scene["robot_anim"]
    num_dofs = len(robot_anim.data.joint_names)
    
    
    env.reset()
    
    while simulation_app.is_running():
        with torch.inference_mode():
            action = torch.zeros((env.num_envs, num_dofs), device=env.device)
            env.step(action)
            
    env.close()
    
if __name__ == "__main__":
    main()
    simulation_app.close()
            
    