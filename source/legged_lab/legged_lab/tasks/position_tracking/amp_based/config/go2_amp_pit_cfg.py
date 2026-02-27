# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .go2_amp_rough_cfg import *
from legged_lab.tasks.position_tracking.gait_reward_based.terrain import PIT_CFG

@configclass
class Go2AmpPitEnvCfg(Go2AmpRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        self.scene.terrain.terrain_generator = PIT_CFG
        self.commands.target_position.min_dist = 2.0
        
        self.rewards.base_lin_vel_z.weight = -0.1
        self.rewards.base_ang_vel_xy.weight = -0.05
        self.rewards.feet_edge.weight = -4.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "Go2AmpPitEnvCfg":
            self.disable_zero_weight_rewards()

@configclass
class Go2AmpPitEnvCfg_PLAY(Go2AmpRoughEnvCfg_PLAY):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = PIT_CFG
