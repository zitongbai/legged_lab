"""
Convert a single GMR motion file to Legged Lab format (.npz).

What does this script do?
    - Reorders DOF data from GMR (MuJoCo order) to Legged Lab (Isaac Lab order).
    - Runs Isaac Lab in render-only mode to collect full-body kinematic data.
    - Computes velocities via central finite differences.
    - Saves result as .npz with all body states.

Usage:
    python scripts/tools/retarget/single_retarget.py \
        --robot g1 \
        --input_file <path_to_gmr_file.pkl> \
        --output_file <path_to_output_file.npz> \
        --config_file scripts/tools/retarget/config/g1_29dof.yaml \
        [--frame_range START END] \
        [--headless] \
        [--device {cpu,cuda:0}]

GMR Format (input):
    pickle file with keys: fps, root_pos, root_rot (xyzw), dof_pos

Legged Lab Format (output .npz):
    fps, root_pos, root_rot (wxyz), dof_pos, body_names,
    body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert a single GMR pkl to Legged Lab npz.")
parser.add_argument("--robot", type=str, default="g1", help="Robot name (default: g1).")
parser.add_argument("--input_file", type=str, required=True, help="Path to input GMR pickle file.")
parser.add_argument("--output_file", type=str, required=True, help="Path to output .npz file.")
parser.add_argument("--config_file", type=str, required=True, help="Path to YAML config (gmr_dof_names, lab_dof_names).")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Frame range: START END (both inclusive). If not provided, all frames are used.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import sys
import yaml
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

if args_cli.robot == "g1":
    from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
else:
    raise ValueError(f"Robot {args_cli.robot} not supported.")

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from gmr_to_lab import ReplayMotionsSceneCfg, extract_gmr_data, run_simulator
except ImportError as e:
    print(f"Error importing from gmr_to_lab.py: {e}")
    raise


if __name__ == "__main__":
    with open(args_cli.config_file) as f:
        config = yaml.safe_load(f)

    gmr_dof_names = config["gmr_dof_names"]
    lab_dof_names = config["lab_dof_names"]

    motion_data_dict = extract_gmr_data(
        gmr_file_path=args_cli.input_file,
        gmr_dof_names=gmr_dof_names,
        lab_dof_names=lab_dof_names,
        start_frame=args_cli.frame_range[0] if args_cli.frame_range else 0,
        end_frame=args_cli.frame_range[1] if args_cli.frame_range else -1,
    )

    fps = motion_data_dict["fps"]
    dt = 1.0 / fps

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args_cli.device))
    scene_cfg = ReplayMotionsSceneCfg(
        num_envs=1, env_spacing=3.0, robot=ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    )
    scene = InteractiveScene(scene_cfg)

    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
    sim.reset()
    print("Simulation starting ...")

    motion_data_dicts = run_simulator(simulation_app, sim, scene, [motion_data_dict])
    motion_data_dict = motion_data_dicts[0]

    print("\n" + "=" * 60)
    print("💾 SAVING CONVERTED DATA")
    print("=" * 60)
    print(f"📁 Output File: {args_cli.output_file}")
    num_frames = args_cli.frame_range[1] - args_cli.frame_range[0] if args_cli.frame_range else "All"
    print(f"🧮 Number of Frames: {num_frames}")
    print(f"🦴 Body count: {len(motion_data_dict['body_names'])}")
    print("=" * 60 + "\n")

    np.savez_compressed(args_cli.output_file, **motion_data_dict)
    print("✅ Data saved successfully.")
    print("=" * 60 + "\n")

    print("Closing simulation app...")
    simulation_app.close()
    print("✅ Simulation app closed.")
