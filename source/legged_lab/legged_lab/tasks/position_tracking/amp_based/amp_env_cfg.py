import math
from dataclasses import MISSING
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_lab.tasks.position_tracking.gait_reward_based.mdp as mdp
import legged_lab.tasks.position_tracking.amp_based.mdp as mdp_amp
##
# Pre-defined configs
##
from legged_lab.tasks.position_tracking.gait_reward_based.terrain import *  # isort: skip
from legged_lab.tasks.position_tracking.gait_reward_based.position_env_cfg import MySceneCfg
from legged_lab.envs import ManagerBasedAmpEnvCfg
from legged_lab.managers import AnimationTermCfg as AnimTerm
from legged_lab.managers import MotionDataTermCfg as MotionDataTerm

@configclass
class AmpSceneCfg(MySceneCfg):
    """Configuration for the terrain scene with a legged robot."""
    # robots
    # robot animation (for reference)
    robot_anim: ArticulationCfg = None
    
##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    target_position = mdp.TerrainBasedPoseCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 6.0),
        simple_heading=True,
        debug_vis=True,
        min_dist=1.0,
        ranges=mdp.TerrainBasedPoseCommandCfg.Ranges(
            pos_x=(-4.5, 4.5), pos_y=(-4.5, 4.5), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        position_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "target_position"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        remaining_time = ObsTerm(
            func=mdp.remaining_time_fraction,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scan = ObsTerm(
            func=mdp.HeightScanRand,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"), 
                    "sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            # self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group. (has privilege observations)"""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        position_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "target_position"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        remaining_time = ObsTerm(
            func=mdp.remaining_time_fraction,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scan = ObsTerm(
            func=mdp.HeightScanRand,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                    "sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        key_body_pos_b = ObsTerm(
            func=mdp_amp.key_body_pos_b,
            params=MISSING,
        )

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True
    
    critic: CriticCfg = CriticCfg()
    
    @configclass
    class DiscriminatorCfg(ObsGroup):
        root_local_rot_tan_norm = ObsTerm(func=mdp_amp.root_local_rot_tan_norm)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        key_body_pos_b = ObsTerm(
            func=mdp_amp.key_body_pos_b,
            params=MISSING,
        )
        feet_contact = ObsTerm(
            func=mdp_amp.feet_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "threshold": 1.0,
            },
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
            self.history_length = 10
            self.flatten_history_dim = False
            
    disc: DiscriminatorCfg = DiscriminatorCfg()
            
    @configclass
    class DiscriminatorDemoCfg(ObsGroup):
        ref_root_local_rot_tan_norm = ObsTerm(
            func=mdp_amp.ref_root_local_rot_tan_norm,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            }
        )
        ref_root_ang_vel_b = ObsTerm(
            func=mdp_amp.ref_root_ang_vel_b,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            }
        )
        ref_joint_pos = ObsTerm(
            func=mdp_amp.ref_joint_pos,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            }
        )
        ref_joint_vel = ObsTerm(
            func=mdp_amp.ref_joint_vel,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            }
        )
        ref_key_body_pos_b = ObsTerm(
            func=mdp_amp.ref_key_body_pos_b,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            }
        )
        ref_feet_contact = ObsTerm(
            func=mdp_amp.ref_feet_contact,
            params={
                "animation": MISSING,
                "height_threshold": 0.03,
                "flatten_steps_dim": False,
            },
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
    
    disc_demo: DiscriminatorDemoCfg = DiscriminatorDemoCfg()
        


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )

    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # reset
    randomize_apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
        },
    )

    reset_from_ref = EventTerm(
        func=mdp_amp.reset_from_ref, 
        mode="reset",
        params=MISSING
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # interval
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # General
    is_terminated = RewTerm(func=mdp.is_terminated, weight=0.0)
    joint_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot")})
    
    #command
    heading_command_error_abs = RewTerm(func=mdp.heading_command_error_abs, weight=0.0, 
                                        params={"command_name": "target_position",
                                                "Tr": 1.0})
    
    #base
    base_height = RewTerm(func=mdp.base_height_l1, weight=0.0, params={"sensor_cfg": SceneEntityCfg("height_scanner_base")})
    flat_orientation = RewTerm(func=mdp.flat_orientation_xy, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot")})
    base_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    base_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    base_acc = RewTerm(func=mdp.base_acc, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot")})
    
    # Joint penalties
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_ratio": 1.0},
    )

    # Action penalties
    applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0.0)

    # Contact sensor
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 1.0,
        },
    )

    # Postion-tracking rewards
    position_tracking = RewTerm(
        func=mdp.task_reward,
        weight=3.0,
        params={"command_name": "target_position",  
                "Tr": 1.0,
                },
    )
    exploration = RewTerm(
        func=mdp.exploration_reward,
        weight=1.5,
        params={"command_name": "target_position", 
                "Tr": 1.0,
                },
    )
    stalling_penalty = RewTerm(
        func=mdp.stalling_penalty,
        weight=-1.0,
        params={"command_name": "target_position"},
    )

    # feet rewards
    feet_acc = RewTerm(
        func=mdp.feet_acceleration_penalty,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
        },
    )
    feet_height = RewTerm(
        func=mdp.feet_height_body,
        weight=0.0,
        params={
            "command_name": "target_position",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "target_height": -0.2,
            "dis_threshold": 0.25,
            "heading_threshold": 0.5,
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_1,
        weight=0.0,
        params={
            "command_name": "target_position",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.5,
            "dis_threshold": 0.25,
            "heading_threshold": 0.5,
        },
    )
    feet_edge = RewTerm(
        func=mdp.feet_edge_penalty,
        weight=0.0,
        params={
            "FL_ray_sensor_cfg": SceneEntityCfg("FL_foot_height_scanner"),
            "FR_ray_sensor_cfg": SceneEntityCfg("FR_foot_height_scanner"),
            "RL_ray_sensor_cfg": SceneEntityCfg("RL_foot_height_scanner"),
            "RR_ray_sensor_cfg": SceneEntityCfg("RR_foot_height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "edge_grad_thresh": 0.1,
            "edge_curvature_thresh": 0.05,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # MDP terminations
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # command_resample
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )

    # Contact sensor
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(40.0)},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_pos,
        params={"threshold": 0.5, "asset_cfg": SceneEntityCfg("robot")}
    )
    
    # change_stalling_penalty = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.stalling_penalty.weight",
    #         "modify_fn": mdp.override_value,
    #         "modify_params": {"value": -12.0, "num_steps": 2000*48}
    #     }
    # )
    
    # change_task_reward = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.position_tracking.weight",
    #         "modify_fn": mdp.override_value,
    #         "modify_params": {"value": 20.0, "num_steps": 2000*48}
    #     }
    # )
    
    update_randomize_reset_base = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.randomize_reset_base.params",
            "modify_fn": mdp.override_value,
            "modify_params": {"value": 
                {"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.2), "yaw": (-3.14, 3.14)},
                "velocity_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)}},
                "num_steps": 5000*48}
        }
    )
    
    # change_actuator_gains = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.randomize_actuator_gains.params",
    #         "modify_fn": mdp.override_value,
    #         "modify_params": {"value": 
    #             {"stiffness_distribution_params": (0.5, 2.0),
    #              "damping_distribution_params": (0.5, 2.0)}, 
    #             "num_steps": 8000*48}
    #     }
    # )
    
    # start_apply_external_force_torque = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.randomize_apply_external_force_torque.params",
    #         "modify_fn": mdp.override_value,
    #         "modify_params": {"value":
    #             {"force_range": (-5.0, 5.0),
    #             "torque_range": (-5.0, 5.0)}, 
    #             "num_steps": 10000*48}
    #     }
    # )
    
    # start_push_robot = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.randomize_push_robot.params",
    #         "modify_fn": mdp.override_value,
    #         "modify_params": {"value": 
    #             {"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}, "num_steps": 12000*48}
    #     }
    # )

@configclass
class MotionDataCfg:
    """Motion data settings for the MDP."""
    motion_dataset = MotionDataTerm(
        motion_data_dir="", 
        motion_data_weights={},
    )
    
@configclass
class AnimationCfg:
    """Animation settings for the MDP."""
    animation = AnimTerm(
        motion_data_term="motion_dataset",
        motion_data_components=[
            "root_pos_w",
            "root_quat",
            "root_vel_w",
            "root_ang_vel_w",
            "dof_pos",
            "dof_vel",
            "key_body_pos_b",
        ], 
        num_steps_to_use=10, 
        random_initialize=True,
        random_fetch=True,
        enable_visualization=False,
    )


##
# Environment configuration
##


@configclass
class LocomotionAmpEnvCfg(ManagerBasedAmpEnvCfg):
    """Configuration for the AMP locomotion environment."""

    # scene
    scene: AmpSceneCfg = AmpSceneCfg(num_envs=4096, env_spacing=10.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # Motion data
    motion_data: MotionDataCfg = MotionDataCfg()
    # Animation
    animation: AnimationCfg = AnimationCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 6.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)
