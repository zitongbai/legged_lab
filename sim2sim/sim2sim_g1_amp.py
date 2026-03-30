# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import torch
import cv2
import matplotlib.pyplot as plt
from pynput import keyboard
import time
from collections import deque


class CommandState:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0
    vx_increment = 0.1
    vy_increment = 0.1
    dyaw_increment = 0.1

    min_vx = -1
    max_vx = 2.5
    min_vy = -0.8
    max_vy = 0.8
    min_dyaw = -1.5
    max_dyaw = 1.5
    camera_follow = True
    reset_requested = False
    
    @classmethod
    def update_vx(cls, delta):
        """update forward velocity"""
        cls.vx = np.clip(cls.vx + delta, cls.min_vx, cls.max_vx)
        print(f"vx: {cls.vx:.2f}, vy: {cls.vy:.2f}, dyaw: {cls.dyaw:.2f}")
    
    @classmethod
    def update_vy(cls, delta):
        """update lateral velocity"""
        cls.vy = np.clip(cls.vy + delta, cls.min_vy, cls.max_vy)
        print(f"vx: {cls.vx:.2f}, vy: {cls.vy:.2f}, dyaw: {cls.dyaw:.2f}")
    
    @classmethod
    def update_dyaw(cls, delta):
        """update angular velocity"""
        cls.dyaw = np.clip(cls.dyaw + delta, cls.min_dyaw, cls.max_dyaw)
        print(f"vx: {cls.vx:.2f}, vy: {cls.vy:.2f}, dyaw: {cls.dyaw:.2f}")

    @classmethod
    def toggle_camera_follow(cls):
        cls.camera_follow = not cls.camera_follow
        print(f"Camera follow: {cls.camera_follow}")
    
    @classmethod
    def reset(cls):
        """reset all velocities to zero"""
        cls.vx = 0.0
        cls.vy = 0.0
        cls.dyaw = 0.0
        print(f"Velocities reset: vx: {cls.vx:.2f}, vy: {cls.vy:.2f}, dyaw: {cls.dyaw:.2f}")

def on_press(key):
    """Key press event handler"""
    try:
        if hasattr(key, 'char') and key.char is not None:
            c = key.char.lower()
            if c == '8':
                CommandState.update_vx(CommandState.vx_increment)
            elif c == '2':
                CommandState.update_vx(-CommandState.vx_increment)
            elif c == '4':
                CommandState.update_vy(CommandState.vy_increment)
            elif c == '6':
                CommandState.update_vy(-CommandState.vy_increment)
            elif c == '7':
                CommandState.update_dyaw(CommandState.dyaw_increment)
            elif c == '9':
                CommandState.update_dyaw(-CommandState.dyaw_increment)
            elif c == 'f':
                CommandState.toggle_camera_follow()
            elif c == '0':
                CommandState.reset_requested = True
                print('Reset requested (0 key pressed)')
    except AttributeError:
        pass

def on_release(key):
    """Key release event handler"""
    pass

def start_keyboard_listener():
    """Start keyboard listener"""
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener

def calc_root_local_rot_tan_norm(quat_mj):
    """
    Args:
        quat_mj: MuJoCo quaternion [w, x, y, z]
    """
    quat_scipy = np.array([quat_mj[1], quat_mj[2], quat_mj[3], quat_mj[0]])
    r = R.from_quat(quat_scipy)
    
    matrix = r.as_matrix()
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])

    yaw_r = R.from_euler('z', yaw)

    root_quat_local_r = yaw_r.inv() * r
    
    root_rotm_local = root_quat_local_r.as_matrix()
    tan_vec = root_rotm_local[:, 0]  # x-axis
    norm_vec = root_rotm_local[:, 2] # z-axis
    
    return np.concatenate([tan_vec, norm_vec])

def key_body_pos_b(data, body_names, base_body_name="pelvis"):
    """
    Compute body positions in base frame using vectorized operations.
    
    Args:
        data: MuJoCo data structure
        body_names: List of body names to extract positions
        base_body_name: Base body name (default "pelvis")
    
    Returns:
        key_body_pos_b: Flattened array of positions in base frame
    """
    root_id = data.model.body(base_body_name).id
    
    root_pos_w = data.xpos[root_id].copy()
    root_quat = data.xquat[root_id][[1, 2, 3, 0]].copy()

    key_body_ids = [data.model.body(name).id for name in body_names]
    key_body_pos_w = np.array([data.xpos[body_id].copy() for body_id in key_body_ids])
    
    rel_pos = key_body_pos_w - root_pos_w

    r = R.from_quat(root_quat)
    key_body_pos_b = r.apply(rel_pos, inverse=True)

    return key_body_pos_b.reshape(-1)

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[:].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    rot_6d = calc_root_local_rot_tan_norm(quat)
    key_pos = key_body_pos_b(
        data,
        body_names=[
            "left_ankle_roll_link", "right_ankle_roll_link",
            "left_wrist_yaw_link", "right_wrist_yaw_link",
            "left_shoulder_roll_link", "right_shoulder_roll_link"
        ]
    )
    return (q, dq, quat, v, omega, gvec, rot_6d, key_pos)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg, headless=False):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.
        headless: If True, run without GUI and save video.

    Returns:
        None
    """
    # Start keyboard listener
    print("=" * 60)
    print("Keyboard control instructions:")
    print("  8 key: Increase forward speed (vx)")
    print("  2 key: Decrease forward speed (vx)")
    print("  4 key: Increase left speed (vy)")
    print("  6 key: Decrease left speed (vy)")
    print("  7 key: Increase left turn rate (dyaw)")
    print("  9 key: Decrease left turn rate (dyaw)")
    print("  0 key: Reset all speeds to 0")
    print("  F key: Toggle camera follow mode")
    print("=" * 60)
    keyboard_listener = start_keyboard_listener()
    
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    data.qpos[-cfg.robot_config.num_actions:] = cfg.robot_config.default_pos
    mujoco.mj_step(model, data)

    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()
    
    if headless:
        renderer = mujoco.Renderer(model, width=1920, height=1080)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
     
        cam = mujoco.MjvCamera()
        cam.distance = 4.0      
        cam.azimuth = 45.0   
        cam.elevation = -20.0   
        cam.lookat = [0, 0, 1]  
        out = cv2.VideoWriter('simulation.mp4', fourcc, 1.0/cfg.sim_config.dt/cfg.sim_config.decimation, (1920, 1080))
    else:
        mode = 'window'
        viewer = mujoco_viewer.MujocoViewer(model, data, mode=mode, width=1920, height=1080)
        
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat = [0, 0, 1]


    target_pos = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
    action = np.zeros((cfg.robot_config.num_actions), dtype=np.double)

    # Term-based history stacking: store each term separately
    hist_terms = {
        "omega": deque(maxlen=cfg.robot_config.frame_stack),     # (3,) per frame
        "rot_6d": deque(maxlen=cfg.robot_config.frame_stack),    # (6,) per frame
        "cmd": deque(maxlen=cfg.robot_config.frame_stack),       # (3,) per frame
        "q_obs": deque(maxlen=cfg.robot_config.frame_stack),     # (29,) per frame
        "dq_obs": deque(maxlen=cfg.robot_config.frame_stack),    # (29,) per frame
        "action": deque(maxlen=cfg.robot_config.frame_stack),    # (29,) per frame
        "key_pos": deque(maxlen=cfg.robot_config.frame_stack),   # (18,) per frame
    }

    low_level_step_count = 0

    time_data = []
    commanded_joint_pos_data = []
    actual_joint_pos_data = []
    tau = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
    tau_data = []
    commanded_lin_vel_x_data = []
    commanded_lin_vel_y_data = []
    commanded_ang_vel_z_data = []
    actual_lin_vel_data = []
    actual_ang_vel_data = []
    
    start_time = time.time()
    
    for step in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        if CommandState.reset_requested:
            print('Performing reset: restoring qpos/qvel and zeroing commands')
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            CommandState.reset()
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            CommandState.reset_requested = False

        # Obtain an observation
        q, dq, quat, v, omega, gvec, rot_6d, key_pos = get_obs(data)
        q = q[-cfg.robot_config.num_actions:]
        dq = dq[-cfg.robot_config.num_actions:]

        if low_level_step_count % cfg.sim_config.decimation == 0:
            q_obs = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
            dq_obs = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
            joint_pos = q
            for i in range(cfg.robot_config.num_actions):
                q_obs[i] = joint_pos[cfg.robot_config.urdf2usd[i]]
                dq_obs[i] = dq[cfg.robot_config.urdf2usd[i]]

            obs = np.zeros([1, cfg.robot_config.num_observations], dtype=np.float32)
            obs[0, 0:3] = omega  # base_ang_vel
            obs[0, 3:9] = rot_6d  # root_local_rot_tan_norm
            obs[0, 9] = CommandState.vx
            obs[0, 10] = CommandState.vy
            obs[0, 11] = CommandState.dyaw
            obs[0, 12:41] = q_obs
            obs[0, 41:70] = dq_obs
            obs[0, 70:99] = action
            obs[0, 99:117] = key_pos
            print("current command: lin vel x={:.2f}, lin vel y={:.2f}, ang vel z={:.2f}".format(CommandState.vx, CommandState.vy, CommandState.dyaw))
            print("current velocity: lin vel x={:.2f}, lin vel y={:.2f}, ang vel z={:.2f}".format(v[0], v[1], omega[2]))

            hist_terms["omega"].append(omega.copy())
            hist_terms["rot_6d"].append(rot_6d.copy())
            hist_terms["cmd"].append(np.array([CommandState.vx, CommandState.vy, CommandState.dyaw]))
            hist_terms["q_obs"].append(q_obs.copy())
            hist_terms["dq_obs"].append(dq_obs.copy())
            hist_terms["action"].append(action.copy())
            hist_terms["key_pos"].append(key_pos.copy())

            policy_input_list = []
            term_dims = {
                "omega": 3,
                "rot_6d": 6,
                "cmd": 3,
                "q_obs": 29,
                "dq_obs": 29,
                "action": 29,
                "key_pos": 18,
            }
            
            for term_name in ["omega", "rot_6d", "cmd", "q_obs", "dq_obs", "action", "key_pos"]:
                term_history = np.array(list(hist_terms[term_name]))
                term_dim = term_dims[term_name]
                
                if len(term_history) < cfg.robot_config.frame_stack:
                    padding_needed = cfg.robot_config.frame_stack - len(term_history)
                    oldest_frame = term_history[0:1] if len(term_history) > 0 else np.zeros((1, term_dim))
                    padding = np.repeat(oldest_frame, padding_needed, axis=0)
                    term_history = np.vstack((padding, term_history))
                
                policy_input_list.append(term_history.flatten())
            
            policy_input = np.concatenate(policy_input_list).reshape(1, -1).astype(np.float32)

            with torch.inference_mode():
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()

            target_q = action * cfg.robot_config.action_scale
            for i in range(cfg.robot_config.num_actions):
                target_pos[i] = target_q[cfg.robot_config.usd2urdf[i]]
            target_pos = target_pos + cfg.robot_config.default_pos

            q_low_freq = q.copy()
            v_low_freq = v[:2].copy()
            omega_low_freq = omega[2].copy()

            time_data.append(step * cfg.sim_config.dt)
            commanded_joint_pos_data.append(target_pos.copy())
            actual_joint_pos_data.append(q_low_freq)
            tau_data.append(tau.copy())
            commanded_lin_vel_x_data.append(CommandState.vx)
            commanded_lin_vel_y_data.append(CommandState.vy)
            commanded_ang_vel_z_data.append(CommandState.dyaw)
            actual_lin_vel_data.append(v_low_freq)
            actual_ang_vel_data.append(omega_low_freq)

            if headless:
                renderer.update_scene(data, camera=cam)
                if CommandState.camera_follow:
                    base_pos = data.qpos[0:3].tolist()
                    cam.lookat = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
                img = renderer.render() 
                out.write(img)
            else:
                if CommandState.camera_follow:
                    base_pos = data.qpos[0:3].tolist()
                    viewer.cam.lookat = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
                viewer.render()
            
        target_vel = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
        tau = pd_control(target_pos, q, cfg.robot_config.kps,
                        target_vel, dq, cfg.robot_config.kds)
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        data.ctrl = tau
        mujoco.mj_step(model, data)

        low_level_step_count += 1
        
        # After each simulation step, calculate the elapsed real time and add delay to match simulation time
        elapsed_real_time = time.time() - start_time
        target_sim_time = (step + 1) * cfg.sim_config.dt
        if elapsed_real_time < target_sim_time:
            time.sleep(target_sim_time - elapsed_real_time)

    if headless:
        out.release()
    else:
        viewer.close()
    
    keyboard_listener.stop()

    print("Simulation finished. Generating plots...")

    time_data = np.array(time_data)
    commanded_joint_pos_data = np.array(commanded_joint_pos_data)
    actual_joint_pos_data = np.array(actual_joint_pos_data)
    tau_data = np.array(tau_data)
    commanded_lin_vel_x_data = np.array(commanded_lin_vel_x_data)
    commanded_lin_vel_y_data = np.array(commanded_lin_vel_y_data)
    commanded_ang_vel_z_data = np.array(commanded_ang_vel_z_data)
    actual_lin_vel_data = np.array(actual_lin_vel_data)
    actual_ang_vel_data = np.array(actual_ang_vel_data)


    # Plot 1: Commanded vs Actual Joint Positions
    num_joints = cfg.robot_config.num_actions
    n_cols = 4
    n_rows = (num_joints + n_cols - 1) // n_cols

    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
    axes1 = axes1.flatten()

    joint_names = [f'Joint {i+1}' for i in range(num_joints)]

    for i in range(num_joints):
        ax = axes1[i]
        ax.plot(time_data, commanded_joint_pos_data[:, i], label='Commanded', linestyle='--')
        ax.plot(time_data, actual_joint_pos_data[:, i], label='Actual')
        ax.set_title(joint_names[i])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [rad]")
        ax.legend()
        ax.grid(True)

    for i in range(num_joints, len(axes1)):
        fig1.delaxes(axes1[i])

    fig1.suptitle("Commanded vs Actual Joint Positions", fontsize=16)
    plt.tight_layout()


    # Plot 2: Commanded vs Actual Base Velocities
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes2[0].plot(time_data, commanded_lin_vel_x_data, label='Commanded Vx', linestyle='--')
    axes2[0].plot(time_data, actual_lin_vel_data[:, 0], label='Actual Vx')
    axes2[0].set_title("Base Linear Velocity X")
    axes2[0].set_xlabel("Time [s]")
    axes2[0].set_ylabel("Velocity [m/s]")
    axes2[0].legend()
    axes2[0].grid(True)

    axes2[1].plot(time_data, commanded_lin_vel_y_data, label='Commanded Vy', linestyle='--')
    axes2[1].plot(time_data, actual_lin_vel_data[:, 1], label='Actual Vy')
    axes2[1].set_title("Base Linear Velocity Y")
    axes2[1].set_xlabel("Time [s]")
    axes2[1].set_ylabel("Velocity [m/s]")
    axes2[1].legend()
    axes2[1].grid(True)

    axes2[2].plot(time_data, commanded_ang_vel_z_data, label='Commanded Dyaw', linestyle='--')
    axes2[2].plot(time_data, actual_ang_vel_data, label='Actual Dyaw')
    axes2[2].set_title("Base Angular Velocity Z (Dyaw)")
    axes2[2].set_xlabel("Time [s]")
    axes2[2].set_ylabel("Angular Velocity [rad/s]")
    axes2[2].legend()
    axes2[2].grid(True)

    fig2.suptitle("Commanded vs Actual Base Velocities", fontsize=16)
    plt.tight_layout()

    fig1.savefig("joint_positions.png")
    fig2.savefig("base_velocities.png")

    print("Plots finished.")
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', default=False, help='terrain mode')
    parser.add_argument('--headless', action='store_true', help='Run without GUI and save video')
    args = parser.parse_args()

    class Sim2SimCfg:

        class sim_config:
            mujoco_model_path = f'sim2sim/assets/g1/mjcf/g1.xml'
            sim_duration = 1000000.0
            dt = 0.005
            decimation = 4

        class robot_config:
            kps = np.array([100, 100, 100, 150, 40, 40, 
                            100, 100, 100, 150, 40, 40, 
                            200, 200, 200, 
                            40, 40, 40, 40, 20, 20, 20, 
                            40, 40, 40, 40, 20, 20, 20], dtype=np.double)
            
            kds = np.array([5.0, 5.0, 5.0, 5.0, 2.0, 2.0,
                            5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 
                            6.0, 6.0, 6.0, 
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.double)
            
            default_pos = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
                                    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
                                    0, 0, 0,
                                    0.3, 0.25, 0, 0.97, 0.15, 0, 0,
                                    0.3, -0.25, 0, 0.97, -0.15, 0, 0], dtype=np.double)
            
            tau_limit = 1. * np.array([88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
                                        88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
                                        88.0, 50.0, 50.0,
                                        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
                                        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0], dtype=np.double)
            frame_stack = 5
            num_single_obs = 117
            num_observations = 117
            num_actions = 29
            action_scale = 0.25
            # usd
            isaac_joint_names = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 
                                'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint', 
                                'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint', 
                                'left_knee_joint', 'right_knee_joint', 
                                'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 
                                'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 
                                'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 
                                'left_ankle_roll_joint', 'right_ankle_roll_joint', 
                                'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
                                'left_elbow_joint', 'right_elbow_joint', 
                                'left_wrist_roll_joint', 'right_wrist_roll_joint', 
                                'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 
                                'left_wrist_yaw_joint', 'right_wrist_yaw_joint']
            # urdf
            mjcf_joint_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
                                'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
                                'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
                                'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
                                'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
            
            usd2urdf = []
            urdf2usd = []
            for name in mjcf_joint_names:
                if name in isaac_joint_names:
                    usd2urdf.append(isaac_joint_names.index(name))
                else:
                    raise ValueError(f"Joint name '{name}' in MJCF not found in USD joint names list.")
            print("USD to URDF joint mapping:", usd2urdf)
            for name in isaac_joint_names:
                if name in mjcf_joint_names:
                    urdf2usd.append(mjcf_joint_names.index(name))
                else:
                    raise ValueError(f"Joint name '{name}' in USD not found in MJCF joint names list.")
            print("URDF to USD joint mapping:", urdf2usd)
    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2SimCfg(), args.headless)
