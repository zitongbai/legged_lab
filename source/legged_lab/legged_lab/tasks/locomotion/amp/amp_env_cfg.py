import math
from dataclasses import MISSING

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.envs.manager_based_amp_env_cfg import ManagerBasedAmpEnvCfg
from legged_lab.managers import AnimationTermCfg as AnimTerm
from legged_lab.managers import MotionDataTermCfg as MotionDataTerm
from isaaclab_tasks.utils import PresetCfg


##
# Physics presets
##


@configclass
class AmpRoughPhysicsCfg(PresetCfg):
    """Multi-backend physics preset for the AMP locomotion env.

    Adapted (copied) from IsaacLab v3.0.0-beta2 official velocity example:
    ``isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg.RoughPhysicsCfg``.
    Kept as a local copy (rather than importing from the velocity package) so the AMP
    task stays self-contained. The ``default`` field reproduces the exact PhysX config
    that AMP previously set lazily in ``__post_init__``, so default (PhysX) training
    behavior is unchanged. The ``newton_mjwarp`` field is provided for future rough /
    Newton work and is only selected via the ``presets=newton_mjwarp`` CLI override.
    """

    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=200,
            nconmax=100,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=1,
        debug_mode=False,
        # 1 cm shape margin is the single most important Newton setting for rough
        # terrain — without it, non-Anymal-D robots fail to learn stable contact
        # on triangle-mesh terrain. See isaaclab_newton 0.5.22 changelog.
        default_shape_cfg=NewtonShapeCfg(margin=0.01),
    )
    physx = default


@configclass
class AmpSceneCfg(InteractiveSceneCfg):
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
    # height scanner — None on the base (plane) scene; the rough config populates it with a
    # RayCasterCfg on generator terrain, and the flat config keeps it None. See
    # g1_amp_rough_env_cfg / g1_amp_flat_env_cfg.
    height_scanner: RayCasterCfg = None
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
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.AmpVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=1.0,
        reset_heading_lookahead=0.5,
        debug_vis=True,
        ranges=mdp.AmpVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # # observation terms (order preserved)
        # # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # projected_gravity = ObsTerm(
        #     func=mdp.projected_gravity,
        #     noise=Unoise(n_min=-0.05, n_max=0.05),
        # )
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # actions = ObsTerm(func=mdp.last_action)

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=5, flatten_history_dim=True
        )
        root_local_rot_tan_norm = ObsTerm(
            func=mdp.root_local_rot_tan_norm,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=5,
            flatten_history_dim=True,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=5,
            flatten_history_dim=True,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=5, flatten_history_dim=True
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5), history_length=5, flatten_history_dim=True
        )
        actions = ObsTerm(func=mdp.last_action, history_length=5, flatten_history_dim=True)
        # height_scan is added by the rough config (g1_amp_rough_env_cfg); it uses a
        # shorter per-term history (3) than the proprioceptive terms (5). This only works
        # because the group-level history_length is None below — a non-None group history
        # would override every term's history_length. See __post_init__.
        # key_body_pos_b = ObsTerm(
        #     func=mdp.key_body_pos_b,
        #     params=MISSING,
        #     noise=Unoise(n_min=-0.08, n_max=0.08),
        #     history_length=5,
        #     flatten_history_dim=True,
        # )
        # root_height = ObsTerm(func=mdp.base_pos_z, history_length=5, flatten_history_dim=True)

        def __post_init__(self):
            # history_length is set per-term (all proprio terms use 5) rather than at the
            # group level, so the rough config can give height_scan its own history (3).
            # A non-None group history_length would override every term's setting.
            self.history_length = None
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group. (has privilege observations)"""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=5, flatten_history_dim=True)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=5, flatten_history_dim=True)
        root_local_rot_tan_norm = ObsTerm(
            func=mdp.root_local_rot_tan_norm, history_length=5, flatten_history_dim=True
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=5,
            flatten_history_dim=True,
        )
        joint_pos = ObsTerm(func=mdp.joint_pos, history_length=5, flatten_history_dim=True)
        joint_vel = ObsTerm(func=mdp.joint_vel, history_length=5, flatten_history_dim=True)
        actions = ObsTerm(func=mdp.last_action, history_length=5, flatten_history_dim=True)
        key_body_pos_b = ObsTerm(
            func=mdp.key_body_pos_b,
            params=MISSING,
            history_length=5,
            flatten_history_dim=True,
        )
        # height_scan is added by the rough config with per-term history 3 (see PolicyCfg).

        def __post_init__(self):
            # history set per-term (see PolicyCfg) so the rough config can give height_scan
            # a shorter history than the proprioceptive terms.
            self.history_length = None
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()

    @configclass
    class DiscriminatorCfg(ObsGroup):
        # root_local_rot_tan_norm removed from the discriminator: it encodes the full base
        # orientation (roll/pitch/yaw), which is exactly what diverges most between the flat
        # reference mocap and rough-terrain motion. Keeping it let the discriminator separate
        # agent-vs-demo almost instantly on rough (disc_loss collapses to ~2e-3 within ~2.7k
        # iters), killing the LSGAN style reward before the policy learns to walk. Dropped so
        # the discriminator scores gait (joint_pos/vel + base_ang_vel), not body tilt. Must stay
        # paired with ref_root_local_rot_tan_norm in DiscriminatorDemoCfg (dims must match).
        # root_local_rot_tan_norm = ObsTerm(func=mdp.root_local_rot_tan_norm)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        # key_body_pos_b = ObsTerm(
        #     func=mdp.key_body_pos_b,
        #     params=MISSING,
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
            self.history_length = 10
            self.flatten_history_dim = False

    disc: DiscriminatorCfg = DiscriminatorCfg()

    @configclass
    class DiscriminatorDemoCfg(ObsGroup):
        # Paired with the disc-side removal above (dims must match). Keep commented together.
        # ref_root_local_rot_tan_norm = ObsTerm(
        #     func=mdp.ref_root_local_rot_tan_norm,
        #     params={
        #         "animation": MISSING,
        #         "flatten_steps_dim": False,
        #     },
        # )
        ref_root_ang_vel_b = ObsTerm(
            func=mdp.ref_root_ang_vel_b,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            },
        )
        ref_joint_pos = ObsTerm(
            func=mdp.ref_joint_pos,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            },
        )
        ref_joint_vel = ObsTerm(
            func=mdp.ref_joint_vel,
            params={
                "animation": MISSING,
                "flatten_steps_dim": False,
            },
        )
        # ref_key_body_pos_b = ObsTerm(
        #     func=mdp.ref_key_body_pos_b,
        #     params={
        #         "animation": MISSING,
        #         "flatten_steps_dim": False,
        #     },
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1

    disc_demo: DiscriminatorDemoCfg = DiscriminatorDemoCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_from_ref = EventTerm(func=mdp.reset_from_ref, mode="reset", params=MISSING)

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=MISSING), "threshold": 1.0},
    )
    # Base-height fall check. sensor_cfg=None here uses absolute world-z (flat-terrain default,
    # identical to the stock root_height_below_minimum). The rough config swaps in the
    # height_scanner RayCaster to make it terrain-relative. This is the primary loophole-closer
    # for the "push-up" pose (base held low but no illegal contact) — it measures how far the
    # base has actually sunk, independent of which links touch the ground.
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum_adaptive,
        params={"minimum_height": 0.2, "sensor_cfg": None},
    )
    # bad_orientation disabled (method X): the 60° tilt limit created a loophole — the
    # policy learned a "push-up" pose (torso held off the ground by the hands, tilt kept
    # just under 60°) that dodged both this term and the torso contact term while still
    # collecting AMP style reward. The official velocity example has no orientation
    # termination at all; rely on base_contact (torso_link) + base_height instead.
    # bad_orientation = DoneTerm(
    #     func=mdp.bad_orientation,
    #     params={
    #         "limit_angle": math.radians(60.0),
    #     },
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # None on the base (plane) config; the rough config sets this to
    # CurrTerm(func=mdp.terrain_levels_vel) to enable terrain-difficulty progression.
    terrain_levels: CurrTerm = None


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

    # Simulation settings — shared multi-backend physics preset (PhysX default + Newton MJWarp)
    sim: SimulationCfg = SimulationCfg(physics=AmpRoughPhysicsCfg())
    # scene
    scene: AmpSceneCfg = AmpSceneCfg(num_envs=4096, env_spacing=2.5)
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
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if getattr(self.scene, "height_scanner", None) is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for the
        # terrain generator (increasing difficulty). Mirrors the official velocity example.
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
