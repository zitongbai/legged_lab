import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a quadruped base environment.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab.envs import ManagerBasedRLEnv

from legged_lab.tasks.locomotion.velocity.config.go2.scandots_rough_env_cfg import UnitreeGo2ScandotsRoughEnvCfg


def main():
    env_cfg = UnitreeGo2ScandotsRoughEnvCfg()

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.height_scanner.debug_vis = True

    env = ManagerBasedRLEnv(cfg=env_cfg)

    count = 0
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            if count % 500 == 0:
                count = 0
                obs, _ = env.reset()

                print("-" * 80)
                print("Resetting envs")

                scan = obs["sensor"]
                print("scan shape:", scan.shape)

            action = torch.randn_like(env.action_manager.action)
            obs, rew, terminated, truncated, info = env.step(action)

            count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
