import os

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg
from legged_lab.rsl_rl import RslRlPpoAmpAlgorithmCfg, RslRlAmpCfg, RslRlOnPolicyRunnerAmpCfg, RslRlPpoActorCriticConv2dCfg
from legged_lab import LEGGED_LAB_ROOT_DIR

@configclass
class G1RoughRslRlOnPolicyRunnerAmpCfg(RslRlOnPolicyRunnerAmpCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "g1_amp_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticConv2dCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        conv_layers_params=[
            {"out_channels": 2, "kernel_size": 3, "stride": 2},
            {"out_channels": 4, "kernel_size": 3, "stride": 2},
        ],
        conv_linear_output_size=8,
    )
    motion_file_path = os.path.join(
        LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof_lock_waist", "retargeted_motion.pkl"
    )
    motion_cfg_path = os.path.join(
        LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof_lock_waist", "retargeted.yaml"
    )
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAmp",
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
            replay_buffer_size=100,
            grad_penalty_scale=10.0,
            amp_trunk_weight_decay=1.0e-4,
            amp_linear_weight_decay=1.0e-2,
            amp_learning_rate=1.0e-4,
            amp_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                amp_reward_scale=2.0,
                task_reward_lerp=0.1
            )
        )
    )

@configclass
class G1FlatRslRlOnPolicyRunnerAmpCfg(G1RoughRslRlOnPolicyRunnerAmpCfg):
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    def __post_init__(self):
        super().__post_init__()
        
        self.experiment_name = "g1_amp_flat"
        
        