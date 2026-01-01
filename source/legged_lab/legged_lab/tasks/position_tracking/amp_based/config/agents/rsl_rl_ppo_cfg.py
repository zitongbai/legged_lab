import os

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlSymmetryCfg
from legged_lab.rsl_rl import RslRlPpoAmpAlgorithmCfg, RslRlAmpCfg, RslRlPpoActorCriticConv2dCfg
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.tasks.position_tracking.amp_based.mdp.symmetry import go2

@configclass
class Go2RoughPPORunnerAmpCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"
    num_steps_per_env = 48
    max_iterations = 50000
    save_interval = 200
    experiment_name = "go2_amp_rough"
    obs_groups = {
        "policy": ["policy"], 
        "critic": ["critic"], 
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"]
    }
    # policy = RslRlPpoActorCriticRecurrentCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[512, 256, 128],
    #     actor_obs_normalization=False,
    #     critic_obs_normalization=False,
    #     activation="elu",
    #     rnn_type="lstm",
    #     rnn_hidden_dim=64,
    #     rnn_num_layers=1
    # )
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
    )
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=1.0,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=1.0e-4,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                style_reward_scale=5.0,
                task_style_lerp=0.75
            ),
            loss_type="LSGAN"
        ),
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=go2.compute_symmetric_states,
            use_mirror_loss=True, mirror_loss_coeff=0.1,
        )
    )

@configclass
class Go2PitPPORunnerAmpCfg(Go2RoughPPORunnerAmpCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_amp_pit"
        
@configclass
class Go2GapPPORunnerAmpCfg(Go2RoughPPORunnerAmpCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_amp_gap"