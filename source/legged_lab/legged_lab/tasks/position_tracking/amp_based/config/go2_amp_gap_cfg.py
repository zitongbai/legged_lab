# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .go2_amp_rough_cfg import *
from legged_lab.tasks.position_tracking.gait_reward_based.terrain import GAP_CFG

@configclass
class Go2AmpGapEnvCfg(Go2AmpRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        self.scene.terrain.terrain_generator = GAP_CFG
        self.commands.target_position.min_dist = 2.0
        
        self.rewards.base_lin_vel_z.weight = 0
        self.rewards.feet_edge.weight = -5.0
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "Go2AmpGapEnvCfg":
            self.disable_zero_weight_rewards()

@configclass
class Go2AmpGapEnvCfg_PLAY(Go2AmpRoughEnvCfg_PLAY):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = GAP_CFG
