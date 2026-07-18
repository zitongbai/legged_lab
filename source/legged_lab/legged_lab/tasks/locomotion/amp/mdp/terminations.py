from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject  # runtime class, guarded per v3 pattern
    from isaaclab.sensors import RayCaster  # runtime class, guarded per v3 pattern
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_below_minimum_adaptive(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    terrain_agg: str = "mean",
) -> torch.Tensor:
    """Terminate when the base height drops below ``minimum_height``.

    Mode is selected by ``sensor_cfg``:

    * ``sensor_cfg=None`` -- absolute world-z height. Identical to the stock
      :func:`isaaclab.envs.mdp.terminations.root_height_below_minimum` (flat terrain only).
    * ``sensor_cfg`` given -- height of the base above the terrain sampled by a RayCaster
      height scanner, so the check is terrain-relative and valid on rough terrain.

    ``terrain_agg`` selects how the scanned terrain height is reduced across rays (only used
    when ``sensor_cfg`` is given), always ignoring missed rays (``ray_hits_w`` holds ``inf``):

    * ``"mean"`` (default) -- mean world-z of the hit points. Mirrors the terrain adjustment in
      the stock :func:`isaaclab.envs.mdp.rewards.base_height_l2` reward.
    * ``"max"`` -- highest hit point under the scanner. Use on stepping-stones-style terrain
      where the scanner footprint straddles stone tops and deep holes: the mean gets dragged
      down by the hole floors (a solid surface at ``holes_depth``, not a miss), which inflates
      the computed clearance and silently disables the fall check. The max tracks the stone the
      robot is actually standing on and is immune to the holes.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        hits_z = sensor.data.ray_hits_w.torch[..., 2]  # (N, B), inf for missed rays
        valid = torch.isfinite(hits_z)
        if terrain_agg == "max":
            # masked max over rays; missed rays -> -inf so they never win. An all-miss row
            # stays -inf, making clearance +inf (never terminates) rather than misfiring.
            terrain_z = hits_z.masked_fill(~valid, float("-inf")).max(dim=1).values
        elif terrain_agg == "mean":
            # masked mean over rays; clamp guards the (degenerate) all-miss row against div-by-zero
            terrain_z = hits_z.masked_fill(~valid, 0.0).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            raise ValueError(f"terrain_agg must be 'mean' or 'max', got {terrain_agg!r}")
    else:
        terrain_z = 0.0

    return asset.data.root_pos_w.torch[:, 2] - terrain_z < minimum_height
