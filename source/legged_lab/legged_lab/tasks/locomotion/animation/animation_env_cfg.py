from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import legged_lab.tasks.locomotion.animation.mdp as mdp
from legged_lab.envs import ManagerBasedAnimationEnvCfg
from legged_lab.managers import AnimationTermCfg as AnimTerm
from legged_lab.managers import MotionDataTermCfg as MotionDataTerm

##
# Scene definition
##


@configclass
class AnimSceneCfg(InteractiveSceneCfg):
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
    # # robots
    # robot: ArticulationCfg = MISSING
    # robots for animation
    robot_anim: ArticulationCfg = MISSING

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

    # It is not used, just a placeholder to comply with the structure.
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot_anim", joint_names=[".*"], scale=0.0, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class AnimCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_anim")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    anim: AnimCfg = AnimCfg()


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    motion_data_finish = DoneTerm(func=mdp.motion_data_finish)


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
            "dof_pos",
            "key_body_pos_b",
        ],
        num_steps_to_use=1,
        random_initialize=True,
        random_fetch=False,
        enable_visualization=True,
    )


##
# Environment configuration
##


@configclass
class AnimationEnvCfg(ManagerBasedAnimationEnvCfg):
    """Configuration for the manager based animation environment."""

    scene: AnimSceneCfg = AnimSceneCfg(num_envs=4096, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

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
