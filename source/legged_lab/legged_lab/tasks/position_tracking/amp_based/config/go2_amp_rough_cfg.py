import os
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from legged_lab.tasks.position_tracking.amp_based.amp_env_cfg import LocomotionAmpEnvCfg
from legged_lab import LEGGED_LAB_ROOT_DIR

##
# Pre-defined configs
##
# use local assets
from legged_lab.assets.unitree import UNITREE_GO2_CFG  # isort: skip

# For Go2 AMP, a common choice is the 4 feet end-effectors.
# The order matters for symmetry augmentation in amp_based.mdp.symmetry.go2.
KEY_BODY_NAMES = [
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
]
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 4


@configclass
class Go2AmpRoughEnvCfg(LocomotionAmpEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"
    # fmt: off
    joint_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        super().__post_init__()
        
        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------------------------------
        # motion data
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "go2", "amp"
        )
        self.motion_data.motion_dataset.motion_data_weights = {}

        # ------------------------------------------------------
        # animation
        # ------------------------------------------------------
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # -----------------------------------------------------
        # Observations
        # -----------------------------------------------------
        
        # critic observations
        
        self.observations.critic.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot", 
                body_names=KEY_BODY_NAMES, 
                preserve_order=True
            )
        }
        
        # discriminator observations
        
        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot", 
                body_names=KEY_BODY_NAMES, 
                preserve_order=True
            )
        }
        self.observations.disc.history_length = AMP_NUM_STEPS
        
        # discriminator demostration observations
        
        self.observations.disc_demo.ref_root_local_rot_tan_norm.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_key_body_pos_b.params["animation"] = ANIMATION_TERM_NAME

        self.observations.policy.base_ang_vel.scale = 0.2
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        # self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        # reset from reference animation (AMP)
        self.events.reset_from_ref.params = {
            "animation": ANIMATION_TERM_NAME,
            "height_offset": 0.05,
        }

        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (0.0, 0.2),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        #start after certain steps
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (0, 0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (0, 0)
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (1.0, 1.0)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (1.0, 1.0)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (0, 0), "y": (0, 0)}
        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -400.0
        self.rewards.joint_deviation.weight = -0.25
        
        # Base
        self.rewards.base_height.weight = -10.0
        self.rewards.flat_orientation.weight = -0.5
        self.rewards.base_lin_vel_z.weight = -0.7
        self.rewards.base_ang_vel_xy.weight = -0.05
        self.rewards.base_acc.weight = -5e-4
        
        # Command
        self.rewards.heading_command_error_abs.weight = 3.0

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2e-4
        self.rewards.joint_vel_l2.weight = -1e-4
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.joint_vel_limits.weight = -1.0
        
        # Action penalties
        self.rewards.applied_torque_limits.weight = -0.2
        self.rewards.action_rate_l2.weight = -2e-5

        # Contact sensor
        self.rewards.undesired_contacts.weight = -2.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_hip", ".*_thigh", ".*_calf"]
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        

        # Position-tracking rewards
        self.rewards.position_tracking.weight = 15.0
        self.rewards.exploration.weight = 5.0
        self.rewards.stalling_penalty.weight = -5.0

        # Others
        self.rewards.feet_acc.weight = -2e-6
        self.rewards.feet_acc.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -2.0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = -5.0
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.params["target_height"] = -0.22
        self.rewards.feet_height.params["dis_threshold"] = 0.25
        self.rewards.feet_height.params["heading_threshold"] = 0.5
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["dis_threshold"] = 0.25
        self.rewards.feet_air_time.params["heading_threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -2.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "Go2AmpRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, "Head_.*"]
        # self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        self.curriculum.terrain_levels.params["threshold"] = 0.25
        # self.curriculum.command_levels = None

@configclass
class Go2AmpRoughEnvCfg_PLAY(Go2AmpRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 50
        self.scene.env_spacing = 8.0
           # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.randomize_apply_external_force_torque = None
        self.events.reset_from_ref = None
        self.curriculum = None
