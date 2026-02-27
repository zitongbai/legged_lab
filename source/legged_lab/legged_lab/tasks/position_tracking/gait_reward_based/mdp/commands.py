from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands import *

from isaaclab.terrains import TerrainImporter
from isaaclab.terrains.utils import *
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class TerrainBasedPoseCommand(UniformPose2dCommand):
    """Command generator that generates pose commands based on the terrain.

    This command generator samples the position commands from the valid patches of the terrain.
    The heading commands are either set to point towards the target or are sampled uniformly.

    It expects the terrain to have a valid position under the key 'target'.
    """

    cfg: TerrainBasedPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TerrainBasedPoseCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)  

        # obtain the terrain asset
        self.terrain: TerrainImporter = env.scene["terrain"]
        
        # obtain the valid targets from the terrain
        if "target" not in self.terrain.flat_patches:
            raise RuntimeError(
                "The terrain-based command generator requires a valid flat patch under 'target' in the terrain."
                f" Found: {list(self.terrain.flat_patches.keys())}")
        # valid targets: (terrain_level, terrain_type, num_patches, 3)
        self.valid_targets: torch.Tensor = self.terrain.flat_patches["target"]
        
    @property
    def robot_pos_w(self) -> torch.Tensor:
        return self.robot.data.root_pos_w
    
    @property
    def robot_heading_w(self) -> torch.Tensor:
        return self.robot.data.heading_w
    
    @property
    def target_pos_w(self) -> torch.Tensor:
        return self.pos_command_w
    
    @property
    def robot_velocity_w(self) -> torch.Tensor:
        return self.robot.data.root_lin_vel_w
    
    @property
    def target_heading_b(self) -> torch.Tensor:
        return self.heading_command_b
    
    # @property
    # def command(self) -> torch.Tensor:
    #     """The desired 2D-pose in base frame. Shape is (num_envs, 3)."""
    #     return self.pos_command_b
        

    def _resample_command(self, env_ids: Sequence[int]):
        # Convert env_ids to a tensor for efficient indexing
        resample_env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Loop until all environments have a valid command
        while len(resample_env_ids) > 0:
            # Number of environments that still need resampling
            num_to_resample = len(resample_env_ids)

            # Sample new position targets only for the environments that need it
            ids = torch.randint(0, self.valid_targets.shape[2], size=(num_to_resample,), device=self.device)
            self.pos_command_w[resample_env_ids] = self.valid_targets[
                self.terrain.terrain_levels[resample_env_ids], self.terrain.terrain_types[resample_env_ids], ids
            ]

            # Offset the position command by the current root height
            self.pos_command_w[resample_env_ids, 2] += self.robot.data.default_root_state[resample_env_ids, 2]

            # Check if the newly sampled positions are too close
            dists = torch.norm(
                self.pos_command_w[resample_env_ids] - self.robot.data.root_pos_w[resample_env_ids], dim=1
            )

            # Identify which environments failed the distance check
            failed_mask = dists < self.cfg.min_dist
            
            # Update the list of environments to resample for the next iteration.
            # Only keep the IDs of the environments that failed.
            resample_env_ids = resample_env_ids[failed_mask]

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            # flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            # # compute errors to find the closest direction to the current heading
            # # this is done to avoid the discontinuity at the -pi/pi boundary
            # curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
            # curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()

            # # set the heading command to the closest direction
            # self.heading_command_w[env_ids] = torch.where(
            #     curr_to_target < curr_to_flipped_target,
            #     target_direction,
            #     flipped_target_direction,
            # )
            self.heading_command_w[env_ids] = target_direction
        else:
            # random heading command
            r = torch.empty(len(env_ids), device=self.device)
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            

@configclass
class TerrainBasedPoseCommandCfg(UniformPose2dCommandCfg):
    """Configuration for the terrain-based position command generator."""

    class_type = TerrainBasedPoseCommand

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""
        
        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the sampled commands."""
    min_dist: float = 1.0
    """Minimum distance between the current robot position and the sampled position command."""
