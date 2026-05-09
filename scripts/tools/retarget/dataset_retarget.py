"""
Batch retargeting tool: convert multiple GMR motion files to Legged Lab format (.npz).

Behavior:
 - Reads all .pkl files from the input directory (sorted).
 - For each file, loads the GMR pickle and reorders DOFs via extract_gmr_data.
 - Runs the simulator once with all motions (num_envs = number of motions).
 - Computes velocities via central finite differences.
 - Saves each converted motion as .npz to the output directory.

Usage example:
    python scripts/tools/retarget/dataset_retarget.py \
        --robot g1 \
        --input_dir data/gmr/ \
        --output_dir data/lab/ \
        --config_file scripts/tools/retarget/config/g1_29dof.yaml

This script does NOT support start/end frame clipping; it converts full motions.
"""

import argparse
import numpy as np
import warnings
import yaml
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Batch retarget GMR -> Legged Lab (multiple files).")
parser.add_argument("--robot", type=str, default="g1", help="Robot name to use (default: g1).")
parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input GMR .pkl files.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to write converted .npz files.")
parser.add_argument(
    "--config_file",
    type=str,
    required=True,
    help="Path to YAML config containing gmr_dof_names and lab_dof_names.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import sys

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


def list_input_files(input_dir: str):
    p = Path(input_dir)
    if not p.exists() or not p.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    return sorted(p.rglob("*.pkl"))


def main():
    with open(args_cli.config_file) as f:
        config = yaml.safe_load(f)

    gmr_dof_names = config["gmr_dof_names"]
    lab_dof_names = config["lab_dof_names"]

    input_files = list_input_files(args_cli.input_dir)
    if len(input_files) == 0:
        print(f"No .pkl files found in input directory: {args_cli.input_dir}")
        return

    Path(args_cli.output_dir).mkdir(parents=True, exist_ok=True)

    input_root = Path(args_cli.input_dir)
    motion_data_dicts = []
    input_rel_paths = []
    fps_values = []

    print(f"Found {len(input_files)} files to convert.")
    for p in input_files:
        print(f"Loading and converting: {p.relative_to(input_root)}")
        motion = extract_gmr_data(
            gmr_file_path=str(p),
            gmr_dof_names=gmr_dof_names,
            lab_dof_names=lab_dof_names,
            start_frame=0,
            end_frame=-1,
        )
        motion_data_dicts.append(motion)
        input_rel_paths.append(p.relative_to(input_root))
        fps_values.append(motion["fps"])

    if not all(f == fps_values[0] for f in fps_values):
        warnings.warn(f"Motions have different fps values: {fps_values}. Using fps from first motion.")

    fps = fps_values[0]
    dt = 1.0 / fps

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args_cli.device))
    scene_cfg = ReplayMotionsSceneCfg(
        num_envs=len(motion_data_dicts),
        env_spacing=3.0,
        robot=ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    )
    scene = InteractiveScene(scene_cfg)

    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
    sim.reset()
    print("Simulation starting ...")

    motion_data_dicts = run_simulator(simulation_app, sim, scene, motion_data_dicts)

    print("Saving converted motions to output directory...")
    for rel_path, motion in zip(input_rel_paths, motion_data_dicts):
        out_path = Path(args_cli.output_dir) / rel_path.with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(out_path), **motion)
        print(f"Saved: {out_path}")

    print("Closing simulation app...")
    simulation_app.close()
    print("Done.")


if __name__ == "__main__":
    main()
