"""
This module provides functionality to convert motion data from GMR format to Legged Lab format.

Ref:
    - https://github.com/xbpeng/MimicKit/blob/main/tools/gmr_to_mimickit/gmr_to_mimickit.py
    - https://github.com/HybridRobotics/whole_body_tracking/blob/main/scripts/csv_to_npz.py

GMR Format:
    The input GMR format should be a pickle file containing a dictionary with keys:
    - 'fps': Frame rate (int)
    - 'root_pos': Root position array, shape (num_frames, 3)
    - 'root_rot': Root rotation quaternions, shape (num_frames, 4), format (x, y, z, w)
    - 'dof_pos': Degrees of freedom positions, shape (num_frames, num_dofs)
    - 'local_body_pos': Currently unused (can be None)
    - 'link_body_list': Currently unused (can be None)

Output Legged Lab Format (.npz):
    - 'fps': Frame rate (int)
    - 'root_pos': Root position, shape (num_frames, 3), world frame, relative to env_origin
    - 'root_rot': Root quaternion, shape (num_frames, 4), format (w, x, y, z)
    - 'dof_pos': DOF positions, shape (num_frames, num_dofs), Isaac Lab joint order
    - 'body_names': Body names array, shape (num_bodies,)
    - 'body_pos_w': All body positions, shape (num_frames, num_bodies, 3), relative to env_origin
    - 'body_quat_w': All body quaternions, shape (num_frames, num_bodies, 4), format (w, x, y, z)
    - 'body_lin_vel_w': All body linear velocities, shape (num_frames, num_bodies, 3), world frame
    - 'body_ang_vel_w': All body angular velocities, shape (num_frames, num_bodies, 3), world frame
"""

import numpy as np
import pickle
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from legged_lab.utils.math import ang_vel_from_quat_diff, vel_forward_diff


def extract_gmr_data(
    gmr_file_path: str,
    gmr_dof_names: list[str],
    lab_dof_names: list[str],
    start_frame: int = 0,
    end_frame: int = -1,
):
    with open(gmr_file_path, "rb") as f:
        gmr_data = pickle.load(f)

    fps = gmr_data["fps"]
    root_pos = gmr_data["root_pos"]        # (num_frames, 3)
    root_rot_quat = gmr_data["root_rot"]   # (num_frames, 4), xyzw
    dof_pos = gmr_data["dof_pos"]          # (num_frames, num_dofs)

    print("\n" + "=" * 60)
    print("📥 LOADED GMR DATA")
    print("=" * 60)
    print(f"⏱️  FPS:           type={type(fps).__name__}, value={fps}")
    print(f"📍 Root Position: type={type(root_pos).__name__}, shape={root_pos.shape}")
    print(f"🔄 Root Rotation: type={type(root_rot_quat).__name__}, shape={root_rot_quat.shape}")
    print(f"🦴 DOF Position:  type={type(dof_pos).__name__}, shape={dof_pos.shape}")
    print("=" * 60 + "\n")

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Expected root_pos shape (num_frames, 3), got {root_pos.shape}")
    if root_rot_quat.ndim != 2 or root_rot_quat.shape[1] != 4:
        raise ValueError(f"Expected root_rot_quat shape (num_frames, 4), got {root_rot_quat.shape}")
    if dof_pos.ndim != 2:
        raise ValueError(f"Expected dof_pos to be 2D array, got {dof_pos.ndim}D")

    num_frames = dof_pos.shape[0]
    if end_frame == -1 or end_frame > num_frames:
        end_frame = num_frames
    assert 0 <= start_frame < end_frame <= num_frames, "Invalid start_frame or end_frame."

    # Reorder DOFs from GMR (MuJoCo) order to Isaac Lab order
    gmr_to_lab_indices = []
    for lab_dof in lab_dof_names:
        if lab_dof in gmr_dof_names:
            gmr_to_lab_indices.append(gmr_dof_names.index(lab_dof))
        else:
            raise ValueError(f"DOF name '{lab_dof}' not found in GMR DOF names.")

    dof_pos_lab = dof_pos[:, gmr_to_lab_indices]

    output_data = {
        "fps": fps,
        "root_pos": root_pos[start_frame:end_frame],
        "root_rot": root_rot_quat[start_frame:end_frame],  # still xyzw, converted in run_simulator
        "dof_pos": dof_pos_lab[start_frame:end_frame],
    }

    return output_data


def _compute_body_velocities(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute body linear and angular velocities via central finite differences.

    Args:
        body_pos_w: Shape (T, B, 3), positions relative to env_origin.
        body_quat_w: Shape (T, B, 4), wxyz quaternions.
        dt: Time step in seconds.

    Returns:
        body_lin_vel_w: Shape (T, B, 3), world-frame linear velocities.
        body_ang_vel_w: Shape (T, B, 3), world-frame angular velocities.
    """
    T, B, _ = body_pos_w.shape

    # Linear velocity via central diff, vectorized over body dimension
    body_lin_vel_w = vel_forward_diff(body_pos_w.reshape(T, -1), dt, method="central").reshape(T, B, 3)

    # Angular velocity: process each body independently
    body_ang_vel_w = torch.zeros(T, B, 3, device=body_pos_w.device)
    for b in range(B):
        body_ang_vel_w[:, b, :] = ang_vel_from_quat_diff(
            body_quat_w[:, b, :], dt, in_frame="world", method="central"
        )

    return body_lin_vel_w, body_ang_vel_w


def run_simulator(
    simulation_app,
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    motion_data_dicts: list[dict[str, np.ndarray]],
):
    robot: Articulation = scene["robot"]

    # Record all body names from the robot model
    all_body_names = list(robot.data.body_names)
    num_bodies = len(all_body_names)

    num_motions = len(motion_data_dicts)
    assert num_motions == scene.num_envs, "Number of motions must match number of environments."

    # Prepare per-motion data tensors
    root_pos_w_list = []
    root_quat_list = []
    dof_pos_list = []
    num_frames_list = []

    for motion_data in motion_data_dicts:
        root_pos_w_list.append(torch.from_numpy(motion_data["root_pos"]).to(scene.device).float())

        # Convert root rotation from xyzw (GMR) to wxyz (Isaac Lab)
        root_quat_tensor = torch.from_numpy(motion_data["root_rot"]).to(scene.device).float()
        root_quat_tensor = math_utils.convert_quat(root_quat_tensor, "wxyz")
        root_quat_tensor = math_utils.quat_unique(root_quat_tensor)
        root_quat_tensor = math_utils.normalize(root_quat_tensor)
        root_quat_list.append(root_quat_tensor)

        dof_pos_list.append(torch.from_numpy(motion_data["dof_pos"]).to(scene.device).float())
        num_frames_list.append(motion_data["dof_pos"].shape[0])

    max_num_frames = max(num_frames_list)

    # Pre-allocate body state collection buffers
    body_pos_w_list = [
        torch.zeros((n, num_bodies, 3), device=scene.device) for n in num_frames_list
    ]
    body_quat_w_list = [
        torch.zeros((n, num_bodies, 4), device=scene.device) for n in num_frames_list
    ]

    count = 0
    dt = sim.cfg.dt

    while simulation_app.is_running():
        root_states = robot.data.default_root_state.clone()
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(robot.data.default_joint_vel)

        for motion_idx in range(num_motions):
            num_frames = num_frames_list[motion_idx]
            frame_idx = count if count < num_frames else num_frames - 1

            root_states[motion_idx, :3] = root_pos_w_list[motion_idx][frame_idx]
            root_states[motion_idx, :3] += scene.env_origins[motion_idx, :3]
            root_states[motion_idx, 3:7] = root_quat_list[motion_idx][frame_idx]
            root_states[motion_idx, 7:13] = 0.0  # velocity unused in render-only mode

            joint_pos[motion_idx, :] = dof_pos_list[motion_idx][frame_idx]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # Render only (no physics step) — kinematics are propagated
        sim.render()
        scene.update(dt)

        # Collect body positions and orientations for each motion
        for motion_idx in range(num_motions):
            if count < num_frames_list[motion_idx]:
                origin = scene.env_origins[motion_idx, :3]
                body_pos_w_list[motion_idx][count] = (
                    robot.data.body_pos_w[motion_idx] - origin.unsqueeze(0)
                )
                body_quat_w_list[motion_idx][count] = robot.data.body_quat_w[motion_idx]

        count += 1
        if count >= max_num_frames:
            break

    print(f"[INFO]: Simulation completed in {count} steps.")

    # Compute velocities via central finite differences and write back into dicts
    for motion_idx, motion_data_dict in enumerate(motion_data_dicts):
        fps = motion_data_dict["fps"]
        motion_dt = 1.0 / fps

        body_pos_w = body_pos_w_list[motion_idx]    # (T, B, 3)
        body_quat_w = body_quat_w_list[motion_idx]  # (T, B, 4)

        body_lin_vel_w, body_ang_vel_w = _compute_body_velocities(body_pos_w, body_quat_w, motion_dt)

        # root_pos and root_rot are taken from body index 0 (pelvis/base_link)
        # to ensure consistency with body_pos_w / body_quat_w
        motion_data_dict["root_pos"] = body_pos_w[:, 0, :].cpu().numpy().astype(np.float32)
        motion_data_dict["root_rot"] = body_quat_w[:, 0, :].cpu().numpy().astype(np.float32)

        motion_data_dict["body_names"] = np.array(all_body_names, dtype=object)
        motion_data_dict["body_pos_w"] = body_pos_w.cpu().numpy().astype(np.float32)
        motion_data_dict["body_quat_w"] = body_quat_w.cpu().numpy().astype(np.float32)
        motion_data_dict["body_lin_vel_w"] = body_lin_vel_w.cpu().numpy().astype(np.float32)
        motion_data_dict["body_ang_vel_w"] = body_ang_vel_w.cpu().numpy().astype(np.float32)

        # dof_pos stays as-is (already reordered by extract_gmr_data)
        motion_data_dict["dof_pos"] = motion_data_dict["dof_pos"].astype(np.float32)

    return motion_data_dicts


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = None
