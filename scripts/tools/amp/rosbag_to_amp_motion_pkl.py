"""Convert ROS2 rosbags (joint_states + IMU) into Legged Lab AMP motion .pkl.

Output format matches MotionDataTerm in legged_lab.managers.motion_data_manager:

* fps: int
* root_pos: (T, 3) float32, world frame
* root_rot: (T, 4) float32, quaternion in (w, x, y, z)
* dof_pos: (T, num_dofs) float32
* key_body_pos: (T, num_key_bodies, 3) float32, world frame
* loop_mode: int (0 = clamp, 1 = wrap)

This script uses Isaac Lab to replay the motion and query key-body world positions
from the Go2 USD, so you don't need to hand-write kinematics.

Example:
	python scripts/tools/amp/rosbag_to_amp_motion_pkl.py \
		--bag_dir /path/to/rosbag2_dir \
		--output_dir source/legged_lab/legged_lab/data/MotionData/go2/amp/walk_and_run \
		--motion_name walk_data_1 \
		--output_fps 50 \
		--loop_mode wrap \
		--headless
"""

from __future__ import annotations

import argparse
import contextlib
import traceback
from pathlib import Path

import joblib
import numpy as np

from isaaclab.app import AppLauncher

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


_TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)


def _read_joint_imu_bag(bag_dir: Path, joint_topic: str, imu_topic: str, joint_num: int) -> dict:
	"""Reads joints and IMU orientation from a ROS2 bag directory."""
	with AnyReader([bag_dir], default_typestore=_TYPESTORE) as reader:
		joint_conns = [c for c in reader.connections if c.topic == joint_topic]
		imu_conns = [c for c in reader.connections if c.topic == imu_topic]

		if not joint_conns:
			raise RuntimeError(f"Bag里没有找到关节话题: {joint_topic}")
		if not imu_conns:
			raise RuntimeError(f"Bag里没有找到IMU话题: {imu_topic}")

		joint_conn = joint_conns[0]
		imu_conn = imu_conns[0]

		joint_time = np.zeros(joint_conn.msgcount, dtype=np.float64)
		joint_pos = np.zeros((joint_conn.msgcount, joint_num), dtype=np.float64)
		joint_names = None
		t0_joint = None

		for i, (_, _, rawdata) in enumerate(reader.messages([joint_conn])):
			msg = reader.deserialize(rawdata, joint_conn.msgtype)
			if i == 0:
				joint_names = list(msg.name)
				if len(joint_names) != joint_num:
					raise ValueError(f"期望{joint_num}个关节，但JointState里有{len(joint_names)}个")
				t0_joint = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

			t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
			joint_time[i] = t - t0_joint
			joint_pos[i, :] = msg.position[:joint_num]

		imu_time = np.zeros(imu_conn.msgcount, dtype=np.float64)
		imu_quat_xyzw = np.zeros((imu_conn.msgcount, 4), dtype=np.float64)
		t0_imu = None

		for i, (_, _, rawdata) in enumerate(reader.messages([imu_conn])):
			msg = reader.deserialize(rawdata, imu_conn.msgtype)
			if i == 0:
				t0_imu = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

			t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
			imu_time[i] = t - t0_imu
			q = msg.orientation
			imu_quat_xyzw[i, :] = [q.x, q.y, q.z, q.w]

	return {
		"joint": {"time": joint_time, "names": joint_names, "position": joint_pos},
		"imu": {"time": imu_time, "orientation_xyzw": imu_quat_xyzw},
	}


def _resample_joints(t_src: np.ndarray, q_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
	q_dst = np.zeros((t_dst.shape[0], q_src.shape[1]), dtype=np.float64)
	for j in range(q_src.shape[1]):
		q_dst[:, j] = np.interp(t_dst, t_src, q_src[:, j])
	return q_dst


def _quat_slerp_batch(q0_wxyz, q1_wxyz, t) -> "torch.Tensor":
	"""Vectorized quaternion SLERP.

	Args:
		q0_wxyz: (N, 4) tensor
		q1_wxyz: (N, 4) tensor
		t: (N,) or (N, 1) tensor in [0, 1]
	"""
	import torch

	if t.ndim == 2 and t.shape[-1] == 1:
		t = t.squeeze(-1)

	# Ensure unit quaternions
	q0 = q0_wxyz / q0_wxyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
	q1 = q1_wxyz / q1_wxyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)

	dot = (q0 * q1).sum(dim=-1)
	# Take the shortest path
	neg_mask = dot < 0.0
	q1 = torch.where(neg_mask.unsqueeze(-1), -q1, q1)
	dot = dot.abs().clamp(-1.0, 1.0)

	theta0 = torch.acos(dot)
	sin_theta0 = torch.sin(theta0)

	# For very small angles, fall back to lerp to avoid division by zero
	eps = 1e-6
	small = sin_theta0.abs() < eps

	# slerp
	theta = theta0 * t
	s0 = torch.sin(theta0 - theta) / sin_theta0.clamp(min=eps)
	s1 = torch.sin(theta) / sin_theta0.clamp(min=eps)
	out = s0.unsqueeze(-1) * q0 + s1.unsqueeze(-1) * q1

	# lerp for small
	lerp = (1.0 - t).unsqueeze(-1) * q0 + t.unsqueeze(-1) * q1
	out = torch.where(small.unsqueeze(-1), lerp, out)

	# Re-normalize
	return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def main() -> None:
	parser = argparse.ArgumentParser(description="Convert ROS2 bag to AMP motion .pkl (Go2).")
	parser.add_argument("--bag_dir", type=str, required=True, help="rosbag2目录(包含metadata.yaml)")
	parser.add_argument("--output_dir", type=str, required=True, help="输出目录(.pkl会写到这里)")
	parser.add_argument("--motion_name", type=str, default=None, help="输出motion文件名(不带扩展名)")

	parser.add_argument("--joint_topic", type=str, default="/joint_states")
	parser.add_argument("--imu_topic", type=str, default="/imu_state_broadcaster/imu")

	parser.add_argument("--output_fps", type=int, default=50, help="输出motion fps")
	parser.add_argument("--root_height", type=float, default=0.35, help="root_pos的z高度(无里程计时使用)")

	parser.add_argument(
		"--loop_mode",
		type=str,
		default="wrap",
		choices=["clamp", "wrap"],
		help="motion循环方式",
	)
	parser.add_argument(
		"--key_body_names",
		nargs="+",
		default=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
		help="需要导出的key-body link名(顺序要和env cfg一致)",
	)

	AppLauncher.add_app_launcher_args(parser)
	args = parser.parse_args()

	bag_dir = Path(args.bag_dir)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	motion_name = args.motion_name or bag_dir.name

	# Launch Isaac Sim
	app_launcher = AppLauncher(args)
	simulation_app = app_launcher.app

	# NOTE:
	# In some Isaac Sim setups, `simulation_app.close()` may raise `SystemExit`,
	# which can unintentionally mask an earlier exception and make the script
	# exit with code 0 without exporting any file. We guard against that below.
	export_path: Path | None = None

	# Imports that require Isaac Sim running
	import torch

	import isaaclab.sim as sim_utils
	import isaaclab.utils.math as math_utils
	from isaaclab.assets import ArticulationCfg, AssetBaseCfg
	from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
	from isaaclab.sim import SimulationContext
	from isaaclab.utils import configclass

	from legged_lab.assets.unitree import UNITREE_GO2_CFG

	@configclass
	class ReplayMotionsSceneCfg(InteractiveSceneCfg):
		ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
		robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

	try:
		bag = _read_joint_imu_bag(bag_dir, args.joint_topic, args.imu_topic, joint_num=12)
		joint_time = bag["joint"]["time"]
		joint_names_in_bag = bag["joint"]["names"]
		joint_pos_in_bag = bag["joint"]["position"]
		imu_time = bag["imu"]["time"]
		imu_quat_xyzw = bag["imu"]["orientation_xyzw"]

		# overlap time window
		t0 = max(float(joint_time[0]), float(imu_time[0]))
		t1 = min(float(joint_time[-1]), float(imu_time[-1]))
		if not (t1 > t0):
			raise RuntimeError("joint和imu时间没有重叠区间，无法对齐")

		fps_out = int(args.output_fps)
		dt_out = 1.0 / float(fps_out)
		t_dst = np.arange(t0, t1, dt_out, dtype=np.float64)
		if t_dst.size < 2:
			raise RuntimeError("有效时长过短，无法生成motion")

		# Go2 joint ordering used by this repo's Go2 env configs
		go2_joint_order = [
			"FL_hip_joint",
			"FR_hip_joint",
			"RL_hip_joint",
			"RR_hip_joint",
			"FL_thigh_joint",
			"FR_thigh_joint",
			"RL_thigh_joint",
			"RR_thigh_joint",
			"FL_calf_joint",
			"FR_calf_joint",
			"RL_calf_joint",
			"RR_calf_joint",
		]
		name_to_idx = {n: i for i, n in enumerate(joint_names_in_bag)}
		reorder_idx = []
		for n in go2_joint_order:
			if n not in name_to_idx:
				raise RuntimeError(f"rosbag joint_states缺少关节: {n}. bag里有: {joint_names_in_bag}")
			reorder_idx.append(name_to_idx[n])

		joint_pos_ordered = joint_pos_in_bag[:, reorder_idx]
		dof_pos = _resample_joints(joint_time, joint_pos_ordered, t_dst).astype(np.float32)

		# IMU quat xyzw -> wxyz
		imu_quat_wxyz = imu_quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32)
		imu_quat_t = torch.from_numpy(imu_quat_wxyz)
		imu_quat_t = math_utils.quat_unique(imu_quat_t)
		imu_quat_t = math_utils.normalize(imu_quat_t)

		imu_time_t = torch.from_numpy(imu_time.astype(np.float32))
		t_dst_t = torch.from_numpy(t_dst.astype(np.float32))

		idx1 = torch.searchsorted(imu_time_t, t_dst_t).clamp(1, imu_time_t.numel() - 1)
		idx0 = idx1 - 1
		tt0 = imu_time_t[idx0]
		tt1 = imu_time_t[idx1]
		alpha = ((t_dst_t - tt0) / (tt1 - tt0).clamp(min=1e-6)).unsqueeze(-1)

		q0 = imu_quat_t[idx0]
		q1 = imu_quat_t[idx1]
		# IsaacLab's `math_utils.quat_slerp` expects a scalar tau; use vectorized SLERP here.
		root_quat = _quat_slerp_batch(q0, q1, alpha)
		root_quat = math_utils.quat_unique(root_quat)
		root_quat = math_utils.normalize(root_quat)
		root_quat_np = root_quat.cpu().numpy().astype(np.float32)

		root_pos = np.zeros((t_dst.shape[0], 3), dtype=np.float32)
		root_pos[:, 2] = float(args.root_height)

		loop_mode = 1 if args.loop_mode == "wrap" else 0

		# Build sim and query key body positions
		sim = SimulationContext(
			sim_utils.SimulationCfg(
				dt=dt_out,
				device="cuda:0" if torch.cuda.is_available() else "cpu",
			)
		)
		scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
		scene = InteractiveScene(scene_cfg)
		sim.reset()

		robot = scene["robot"]
		lab_body_names = robot.data.body_names
		key_body_indices = []
		for n in args.key_body_names:
			if n not in lab_body_names:
				raise RuntimeError(f"Key body '{n}'不在Go2 body_names里。可用body_names: {lab_body_names}")
			key_body_indices.append(lab_body_names.index(n))

		key_body_pos_w = np.zeros((t_dst.shape[0], len(key_body_indices), 3), dtype=np.float32)

		root_states = robot.data.default_root_state.clone()
		joint_pos_t = robot.data.default_joint_pos.clone()
		joint_vel_t = torch.zeros_like(robot.data.default_joint_vel)

		for k in range(t_dst.shape[0]):
			root_states[:, :3] = torch.from_numpy(root_pos[k : k + 1]).to(scene.device)
			root_states[:, 3:7] = torch.from_numpy(root_quat_np[k : k + 1]).to(scene.device)
			root_states[:, 7:13] = 0.0

			joint_pos_t[:, :] = torch.from_numpy(dof_pos[k : k + 1]).to(scene.device)

			robot.write_root_state_to_sim(root_states)
			robot.write_joint_state_to_sim(joint_pos_t, joint_vel_t)

			sim.render()
			scene.update(dt_out)

			pos_w = robot.data.body_pos_w[0, key_body_indices, :] - scene.env_origins[0, :3]
			key_body_pos_w[k, :, :] = pos_w.detach().cpu().numpy().astype(np.float32)

		motion_dict = {
			"fps": int(fps_out),
			"root_pos": root_pos,
			"root_rot": root_quat_np,
			"dof_pos": dof_pos,
			"key_body_pos": key_body_pos_w,
			"loop_mode": int(loop_mode),
		}

		export_path = output_dir / f"{motion_name}.pkl"
		print(f"\n[rosbag_to_amp_motion_pkl] Writing motion to: {export_path}", flush=True)
		joblib.dump(motion_dict, export_path)

		print("\n" + "=" * 60)
		print("✅ AMP motion导出完成")
		print("=" * 60)
		print(f"输出: {export_path}")
		print(f"fps: {motion_dict['fps']}")
		print(f"frames: {motion_dict['dof_pos'].shape[0]}")
		print(f"dofs: {motion_dict['dof_pos'].shape[1]}")
		print(f"key bodies: {motion_dict['key_body_pos'].shape[1]}")
		print("=" * 60)

	except Exception:
		print("\n[rosbag_to_amp_motion_pkl] ERROR: motion export failed.", flush=True)
		if export_path is not None:
			print(f"[rosbag_to_amp_motion_pkl] Intended output: {export_path}", flush=True)
		traceback.print_exc()
		raise

	finally:
		with contextlib.suppress(SystemExit):
			simulation_app.close()


if __name__ == "__main__":
	main()