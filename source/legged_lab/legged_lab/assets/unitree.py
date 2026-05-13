"""Configuration for Unitree robots.

Reference: https://github.com/unitreerobotics/unitree_rl_lab
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets import unitree_actuators


@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Configuration for Unitree articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9


@configclass
class UnitreeUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )


UNITREE_GO2_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/Unitree/go2/usd/go2.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "GO2HV": unitree_actuators.UnitreeActuatorCfg_Go2HV(
            joint_names_expr=[".*"],
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
        ),
    },
    # fmt: off
    joint_sdk_names=[
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
    ],
    # fmt: on
)


UNITREE_G1_29DOF_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/Unitree/g1_29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "N7520-14.3": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            stiffness={
                ".*_hip_.*": 100.0,
                "waist_yaw_joint": 200.0,
            },
            damping={
                ".*_hip_.*": 2.0,
                "waist_yaw_joint": 5.0,
            },
            armature=0.01,
        ),
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
            effort_limit_sim=139,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_hip_roll_.*": 100.0,
                ".*_knee_.*": 150.0,
            },
            damping={
                ".*_hip_roll_.*": 2.0,
                ".*_knee_.*": 4.0,
            },
            armature=0.01,
        ),
        "N5020-16": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*",
                ".*_elbow_.*",
                ".*_wrist_roll.*",
                ".*_ankle_.*",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=25,
            velocity_limit_sim=37,
            stiffness=40.0,
            damping={
                ".*_shoulder_.*": 1.0,
                ".*_elbow_.*": 1.0,
                ".*_wrist_roll.*": 1.0,
                ".*_ankle_.*": 2.0,
                "waist_.*_joint": 5.0,
            },
            armature=0.01,
        ),
        "W4010-25": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
            effort_limit_sim=5,
            velocity_limit_sim=22,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
)


# BeyondMimic (WBT) physical parameters
_BM_ARMATURE_5020 = 0.003609725
_BM_ARMATURE_7520_14 = 0.010177520
_BM_ARMATURE_7520_22 = 0.025101925
_BM_ARMATURE_4010 = 0.00425

_BM_NATURAL_FREQ = 10 * 2.0 * 3.1415926535
_BM_DAMPING_RATIO = 2.0

_BM_STIFFNESS_5020 = _BM_ARMATURE_5020 * _BM_NATURAL_FREQ**2
_BM_STIFFNESS_7520_14 = _BM_ARMATURE_7520_14 * _BM_NATURAL_FREQ**2
_BM_STIFFNESS_7520_22 = _BM_ARMATURE_7520_22 * _BM_NATURAL_FREQ**2
_BM_STIFFNESS_4010 = _BM_ARMATURE_4010 * _BM_NATURAL_FREQ**2

_BM_DAMPING_5020 = 2.0 * _BM_DAMPING_RATIO * _BM_ARMATURE_5020 * _BM_NATURAL_FREQ
_BM_DAMPING_7520_14 = 2.0 * _BM_DAMPING_RATIO * _BM_ARMATURE_7520_14 * _BM_NATURAL_FREQ
_BM_DAMPING_7520_22 = 2.0 * _BM_DAMPING_RATIO * _BM_ARMATURE_7520_22 * _BM_NATURAL_FREQ
_BM_DAMPING_4010 = 2.0 * _BM_DAMPING_RATIO * _BM_ARMATURE_4010 * _BM_NATURAL_FREQ

UNITREE_G1_29DOF_BEYONDMIMIC_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/Unitree/g1_29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": _BM_STIFFNESS_7520_14,
                ".*_hip_roll_joint": _BM_STIFFNESS_7520_22,
                ".*_hip_yaw_joint": _BM_STIFFNESS_7520_14,
                ".*_knee_joint": _BM_STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": _BM_DAMPING_7520_14,
                ".*_hip_roll_joint": _BM_DAMPING_7520_22,
                ".*_hip_yaw_joint": _BM_DAMPING_7520_14,
                ".*_knee_joint": _BM_DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": _BM_ARMATURE_7520_14,
                ".*_hip_roll_joint": _BM_ARMATURE_7520_22,
                ".*_hip_yaw_joint": _BM_ARMATURE_7520_14,
                ".*_knee_joint": _BM_ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * _BM_STIFFNESS_5020,
            damping=2.0 * _BM_DAMPING_5020,
            armature=2.0 * _BM_ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * _BM_STIFFNESS_5020,
            damping=2.0 * _BM_DAMPING_5020,
            armature=2.0 * _BM_ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness=_BM_STIFFNESS_7520_14,
            damping=_BM_DAMPING_7520_14,
            armature=_BM_ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": _BM_STIFFNESS_5020,
                ".*_shoulder_roll_joint": _BM_STIFFNESS_5020,
                ".*_shoulder_yaw_joint": _BM_STIFFNESS_5020,
                ".*_elbow_joint": _BM_STIFFNESS_5020,
                ".*_wrist_roll_joint": _BM_STIFFNESS_5020,
                ".*_wrist_pitch_joint": _BM_STIFFNESS_4010,
                ".*_wrist_yaw_joint": _BM_STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": _BM_DAMPING_5020,
                ".*_shoulder_roll_joint": _BM_DAMPING_5020,
                ".*_shoulder_yaw_joint": _BM_DAMPING_5020,
                ".*_elbow_joint": _BM_DAMPING_5020,
                ".*_wrist_roll_joint": _BM_DAMPING_5020,
                ".*_wrist_pitch_joint": _BM_DAMPING_4010,
                ".*_wrist_yaw_joint": _BM_DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": _BM_ARMATURE_5020,
                ".*_shoulder_roll_joint": _BM_ARMATURE_5020,
                ".*_shoulder_yaw_joint": _BM_ARMATURE_5020,
                ".*_elbow_joint": _BM_ARMATURE_5020,
                ".*_wrist_roll_joint": _BM_ARMATURE_5020,
                ".*_wrist_pitch_joint": _BM_ARMATURE_4010,
                ".*_wrist_yaw_joint": _BM_ARMATURE_4010,
            },
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
)

# Per-joint action scale: 0.25 * effort_limit / stiffness (mirrors WBT G1_ACTION_SCALE)
G1_BEYONDMIMIC_ACTION_SCALE: dict[str, float] = {}
for _actuator in UNITREE_G1_29DOF_BEYONDMIMIC_CFG.actuators.values():
    _effort = _actuator.effort_limit_sim
    _stiffness = _actuator.stiffness
    _names = _actuator.joint_names_expr
    if not isinstance(_effort, dict):
        _effort = {n: _effort for n in _names}
    if not isinstance(_stiffness, dict):
        _stiffness = {n: _stiffness for n in _names}
    for _name in _names:
        if _name in _effort and _name in _stiffness and _stiffness[_name]:
            G1_BEYONDMIMIC_ACTION_SCALE[_name] = 0.25 * _effort[_name] / _stiffness[_name]
