from __future__ import annotations

from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_DATA_DIR
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG
from legged_lab.tasks.locomotion.tracking.tracking_env_cfg import TrackingEnvCfg


KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]

TRACKING_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

END_EFFECTOR_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

MOTION_DATA_WEIGHTS = {
    "B10_-__Walk_turn_left_45_stageii": 1.0,
    "B11_-__Walk_turn_left_135_stageii": 1.0,
    "B13_-__Walk_turn_right_90_stageii": 1.0,
    "B14_-__Walk_turn_right_45_t2_stageii": 1.0,
    "B15_-__Walk_turn_around_stageii": 1.0,
    "B22_-__side_step_left_stageii": 1.0,
    "B23_-__side_step_right_stageii": 1.0,
    "B4_-_Stand_to_Walk_backwards_stageii": 1.0,
    "B9_-__Walk_turn_left_90_stageii": 1.0,
    "C11_-_run_turn_left_90_stageii": 1.0,
    "C12_-_run_turn_left_45_stageii": 1.0,
    "C13_-_run_turn_left_135_stageii": 1.0,
    "C14_-_run_turn_right_90_stageii": 1.0,
    "C15_-_run_turn_right_45_stageii": 1.0,
    "C16_-_run_turn_right_135_stageii": 1.0,
    "C17_-_run_change_direction_stageii": 1.0,
    "C1_-_stand_to_run_stageii": 1.0,
    "C3_-_run_stageii": 1.0,
    "C4_-_run_to_walk_a_stageii": 1.0,
    "C5_-_walk_to_run_stageii": 1.0,
    "C6_-_stand_to_run_backwards_stageii": 1.0,
    "C8_-_run_backwards_to_stand_stageii": 1.0,
    "C9_-_run_backwards_turn_run_forward_stageii": 1.0,
    "Walk_B10_-_Walk_turn_left_45_stageii": 1.0,
    "Walk_B13_-_Walk_turn_right_45_stageii": 1.0,
    "Walk_B15_-_Walk_turn_around_stageii": 1.0,
    "Walk_B16_-_Walk_turn_change_stageii": 1.0,
    "Walk_B22_-_Side_step_left_stageii": 1.0,
    "Walk_B23_-_Side_step_right_stageii": 1.0,
    "Walk_B4_-_Stand_to_Walk_Back_stageii": 1.0,
}


@configclass
class G1TrackingEnvCfg(TrackingEnvCfg):
    """G1 motion tracking environment."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = 0.5

        self.motion_data.motion_dataset.motion_data_dir = f"{LEGGED_LAB_DATA_DIR}/legged_lab/unitree_g1/amp"
        self.motion_data.motion_dataset.key_body_names = list(KEY_BODY_NAMES)
        self.motion_data.motion_dataset.motion_data_weights = dict(MOTION_DATA_WEIGHTS)

        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = list(TRACKING_BODY_NAMES)

        self.events.base_com.params["asset_cfg"].body_names = "torso_link"
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
        ]
        self.terminations.ee_body_pos.params["body_names"] = list(END_EFFECTOR_BODY_NAMES)


@configclass
class G1TrackingEnvCfg_PLAY(G1TrackingEnvCfg):
    """Play configuration for G1 motion tracking."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.commands.motion.debug_vis = True
        self.commands.motion.adaptive_sampling = False
