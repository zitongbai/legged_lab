import argparse
import os

import joblib

parser = argparse.ArgumentParser(description="Print motion names from a motion file.")
parser.add_argument(
    "--robot",
    type=str,
    default="g1_27dof",
    help="The robot to be used.",
)
parser.add_argument(
    "--motion_file",
    type=str,
    default="retargeted_motion.pkl",
    help="File name of the motion data to be loaded.",
)

args = parser.parse_args()

LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
motion_file = os.path.join(LEGGED_LAB_ROOT_DIR, "data", "MotionData", args.robot, args.motion_file)

assert os.path.exists(motion_file), f"Motion file {motion_file} does not exist, please check the file."

motion_dict = joblib.load(motion_file)
motion_data_dict = motion_dict["retarget_data"]
motion_names = list(motion_data_dict.keys())
print("Motion names:")
for name in motion_names:
    print(f'"{name}"')
