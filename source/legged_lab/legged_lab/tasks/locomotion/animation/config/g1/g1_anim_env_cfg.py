import os

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_DATA_DIR

##
# Pre-defined configs
##
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG
from legged_lab.tasks.locomotion.animation.animation_env_cfg import AnimationEnvCfg


@configclass
class G1AnimEnvCfg(AnimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot_anim = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_anim")
        self.scene.robot_anim.spawn.rigid_props.disable_gravity = True  # type: ignore
        self.scene.robot_anim.spawn.articulation_props.enabled_self_collisions = False  # type: ignore
        self.scene.robot_anim.spawn.activate_contact_sensors = False  # type: ignore
        self.scene.robot_anim.spawn.collision_props = sim_utils.CollisionPropertiesCfg(  # type: ignore
            collision_enabled=False
        )

        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_DATA_DIR, "MotionData", "g1_29dof", "deepmimic"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "C4_-_run_to_walk_a_stageii": 1.0,
        }
        # self.motion_data.motion_dataset.key_link_names = [
        #     "left_ankle_roll_link",
        #     "right_ankle_roll_link",
        #     "left_wrist_yaw_link",
        #     "right_wrist_yaw_link",
        # ]

        self.animation.animation.random_initialize = True
        self.animation.animation.num_steps_to_use = 10
