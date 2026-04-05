import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import legged_lab.tasks.locomotion.deepmimic.mdp as mdp
from legged_lab.envs import ManagerBasedAnimationEnvCfg
from legged_lab.managers import AnimationTermCfg as AnimTerm
from legged_lab.managers import MotionDataTermCfg as MotionDataTerm

##
# Scene definition
##


@configclass
class DeepMimicSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # robot animation (for reference)
    robot_anim: ArticulationCfg = None
    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        root_rot_tan_norm = ObsTerm(func=mdp.root_rot_tan_norm)
        root_vel_w = ObsTerm(func=mdp.root_lin_vel_w)
        root_ang_vel_w = ObsTerm(func=mdp.root_ang_vel_w)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        key_body_pos_b = ObsTerm(
            func=mdp.key_body_pos_b,
            params=MISSING,
        )
        root_height = ObsTerm(func=mdp.base_pos_z)

        ref_root_pos_error = ObsTerm(func=mdp.ref_root_pos_error, params=MISSING)
        ref_root_rot_tan_norm = ObsTerm(func=mdp.ref_root_rot_tan_norm, params=MISSING)
        ref_joint_pos = ObsTerm(func=mdp.ref_joint_pos, params=MISSING)
        ref_key_body_pos_b = ObsTerm(func=mdp.ref_key_body_pos_b, params=MISSING)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (-3.0, 3.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    reset_from_ref = EventTerm(func=mdp.reset_from_ref, mode="reset", params=MISSING)


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    ref_track_root_pos_w_error_exp = RewTerm(func=mdp.ref_track_root_pos_w_error_exp, weight=1.0, params=MISSING)

    ref_track_quat_error_exp = RewTerm(func=mdp.ref_track_quat_error_exp, weight=0.5, params=MISSING)

    ref_track_root_vel_w_error_exp = RewTerm(func=mdp.ref_track_root_vel_w_error_exp, weight=0.1, params=MISSING)

    ref_track_root_ang_vel_w_error_exp = RewTerm(
        func=mdp.ref_track_root_ang_vel_w_error_exp, weight=0.1, params=MISSING
    )

    ref_track_key_body_pos_b_error_exp = RewTerm(
        func=mdp.ref_track_key_body_pos_b_error_exp, weight=0.3, params=MISSING
    )

    ref_track_dof_pos_error_exp = RewTerm(func=mdp.ref_track_dof_pos_error_exp, weight=0.5, params=MISSING)

    ref_track_dof_vel_error_exp = RewTerm(func=mdp.ref_track_dof_vel_error_exp, weight=0.1, params=MISSING)

    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-8)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=MISSING), "threshold": 1.0},
    )
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": math.radians(50.0),
        },
    )
    # motion_data_finish = DoneTerm(
    #     func=mdp.motion_data_finish
    # )

    deviation_root_pos_w = DoneTerm(func=mdp.deviation_root_pos_w, params=MISSING)
    # deviation_key_body_pos_b = DoneTerm(
    #     func=mdp.deviation_key_body_pos_b,
    #     params=MISSING
    # )
    deviation_key_body_pos_w = DoneTerm(func=mdp.deviation_key_body_pos_w, params=MISSING)


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
        num_steps_to_use=4,
        random_initialize=True,
        random_fetch=False,
        enable_visualization=False,
    )


##
# Environment configuration
##


@configclass
class DeepMimicEnvCfg(ManagerBasedAnimationEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: DeepMimicSceneCfg = DeepMimicSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    motion_data: MotionDataCfg = MotionDataCfg()
    animation: AnimationCfg = AnimationCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
