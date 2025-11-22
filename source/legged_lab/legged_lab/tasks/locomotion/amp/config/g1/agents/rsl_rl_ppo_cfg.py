import os

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg
from legged_lab.rsl_rl import RslRlPpoAmpAlgorithmCfg, RslRlAmpCfg, RslRlPpoActorCriticConv2dCfg
from legged_lab import LEGGED_LAB_ROOT_DIR

@configclass
class G1RslRlOnPolicyRunnerAmpCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "g1_amp"
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
        lam=0.95,
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
                task_style_lerp=0.5
            ),
        )
    )



