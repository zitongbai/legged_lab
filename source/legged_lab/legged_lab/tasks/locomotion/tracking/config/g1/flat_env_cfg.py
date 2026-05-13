from __future__ import annotations

from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_DATA_DIR
from legged_lab.assets.unitree import G1_BEYONDMIMIC_ACTION_SCALE, UNITREE_G1_29DOF_BEYONDMIMIC_CFG
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
    "Form_1_stageii": 1.0,
}


@configclass
class G1TrackingEnvCfg(TrackingEnvCfg):
    """G1 motion tracking environment."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_G1_29DOF_BEYONDMIMIC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_BEYONDMIMIC_ACTION_SCALE

        self.motion_data.motion_dataset.motion_data_dir = f"{LEGGED_LAB_DATA_DIR}/legged_lab/unitree_g1/AMASS/ACCAD/Male2MartialArtsExtended_c3d/"
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
