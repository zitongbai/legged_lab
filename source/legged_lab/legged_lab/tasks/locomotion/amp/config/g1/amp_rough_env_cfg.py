import os
import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.tasks.locomotion.velocity.velocity_env_cfg import RewardsCfg, MySceneCfg, ScandotsSceneCfg
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg

import isaaclab.terrains as terrain_gen

##
# Pre-defined configs
##
from legged_lab.assets.unitree import G1_29DOF_LOCK_WAIST_MINIMAL_CFG, G1_29DOF_LOCK_WAIST_CFG

from legged_lab import LEGGED_LAB_ROOT_DIR



@configclass
class G1AmpRewards():
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    
    alive = RewTerm(
        func=mdp.is_alive,
        weight=0.1,
    )
    
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                ],
            )
        },
    )
    joint_deviation_wrists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_yaw_joint")},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", 
                body_names=[".*_wrist_yaw_link", ".*_wrist_pitch_link"]
            ), 
            "threshold": 1.0
        },
    )
    feet_orientation_l2 = RewTerm(
        func=mdp.feet_orientation_l2,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", 
                body_names=".*_ankle_roll_link",
            ),
        }
    )


@configclass
class G1AmpRoughEnvCfg(LocomotionAmpEnvCfg):
    rewards: G1AmpRewards = G1AmpRewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # ------------------------------------------------------
        # Scene
        # ------------------------------------------------------
        self.scene.robot = G1_29DOF_LOCK_WAIST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # height scanner
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"
        # terrain
        self.scene.terrain.terrain_generator.sub_terrains = {   # type: ignore
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.2,
                step_height_range=(0.05, 0.23),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.2,
                step_height_range=(0.05, 0.23),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
            ),
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            ),
            "hf_stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
                proportion=0.2,
                stone_height_max=0.0,
                stone_width_range=(0.4, 1.0),
                stone_distance_range=(0.1, 0.3),
                holes_depth=-5.0,
                platform_width=2.0,
                border_width=0.25,
            ),
        }
        self.scene.terrain.terrain_generator.curriculum = True  # type: ignore
        # make sure the curriculum is enabled
        
        # ------------------------------------------------------
        # motion loader
        # ------------------------------------------------------
        self.motion_loader.motion_file_path = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof_lock_waist", "retargeted_motion.pkl"
        )
        self.motion_loader.key_links_mapping = {
            # the keys are the names of the links in the motion dataset
            # the values are the names of the links in lab 
            "left_ankle_roll_link": "left_ankle_roll_link",
            "right_ankle_roll_link": "right_ankle_roll_link",
            "left_rubber_hand": "left_wrist_yaw_link",
            "right_rubber_hand": "right_wrist_yaw_link",
        }
        self.motion_loader.motion_weights = {
            # the motion names can be obtained by running `utils/print_motion_names.py`
            # "36_08_poses": 1.0,     # stairs
            # "36_26_poses": 1.0,     # stairs
            # "36_32_poses": 1.0,     # stairs
            # "20_05_poses": 1.0,      # walk with arm
            # "22_25_poses": 1.0,      # slow walk
            # "10_04_poses": 1.0,      # walk from stand
            "08_09_poses": 1.0,      # walk fast in large step
            "08_03_poses": 1.0,      # walk fast in large step
            "08_04_poses": 1.0,      # walk slow in large step
            # "08_06_poses": 1.0,      # walk fast in large step
            # "08_08_poses": 1.0,      # walk fast in large step
            # "16_34_poses": 1.0,      # slow walk and stop
            "77_02_poses": 1.0,      # stand
            # "82_08_poses": 1.0,      # stand, then slow walk
            # "105_10_poses": 1.0,     # walk and turn 180
            # "105_27_poses": 1.0,     # walk and turn 180
        }
        
        # ------------------------------------------------------
        # Observations
        # ------------------------------------------------------
        # self.observations.policy.base_lin_vel = None
        self.observations.amp.key_links_pos_b.params["local_pos_dict"] = {
            "left_ankle_roll_link": (0.0, 0.0, 0.0),
            "right_ankle_roll_link": (0.0, 0.0, 0.0),
            "left_wrist_yaw_link": (0.0415, 0.003, 0.0),
            "right_wrist_yaw_link": (0.0415, -0.003, 0.0),
            # "waist_yaw_link": (0.0, 0.0, 0.4),
            # "left_shoulder_roll_link": (0.0, 0.0, 0.0),
            # "right_shoulder_roll_link": (0.0, 0.0, 0.0),
            # "left_hip_pitch_link": (0.0, 0.0, 0.0),
            # "right_hip_pitch_link": (0.0, 0.0, 0.0),
            # "left_elbow_link": (0.0, 0.0, 0.0),
            # "right_elbow_link": (0.0, 0.0, 0.0),
            # "left_knee_link": (0.0, 0.0, 0.0),
            # "right_knee_link": (0.0, 0.0, 0.0),
        }
        
        # ------------------------------------------------------
        # Curriculum
        # ------------------------------------------------------
        # self.curriculum.terrain_levels = None   # TODO
        
        # ------------------------------------------------------
        # Events
        # ------------------------------------------------------
        self.events.push_robot = None       # TODO
        self.events.add_base_mass = None    # TODO
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["waist_yaw_link"]
        self.events.reset_base_rsi.params["pos_rsi"] = False    # no offset in x and y for the root position during RSI

        # ------------------------------------------------------
        # Rewards
        # ------------------------------------------------------
        # For AMP, we only needs a few rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        
        self.rewards.termination_penalty.weight = 0.0
        self.rewards.alive.weight = 0.1
        
        self.rewards.dof_pos_limits.weight = -0.0
        self.rewards.joint_deviation_hip.weight = -0.0
        self.rewards.joint_deviation_arms.weight = -0.0
        self.rewards.joint_deviation_wrists.weight = -0.0
        self.rewards.joint_deviation_waist.weight = -0.0
        
        self.rewards.undesired_contacts.weight = -0.1
        
        self.rewards.feet_orientation_l2.weight = 0.0
        
        
        # ------------------------------------------------------
        # Commands
        # ------------------------------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        
        # ------------------------------------------------------
        # terminations
        # ------------------------------------------------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "waist_yaw_link", "pelvis", ".*_shoulder_.*_link", ".*_elbow_link",
        ]
        

@configclass
class G1AmpRoughEnvCfg_PLAY(G1AmpRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        
        self.scene.height_scanner.debug_vis = True


