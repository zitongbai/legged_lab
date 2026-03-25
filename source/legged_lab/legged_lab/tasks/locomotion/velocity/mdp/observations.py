from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def height_scan_ch(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.

    add a channel dimension to the output tensor, so that it can be used as a 2D image

    ref: isaaclab.envs.mdp.observations.height_scan
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    ordering = sensor.cfg.pattern_cfg.ordering
    """Specifies the ordering of points in the generated grid. Defaults to ``"xy"``.

    Consider a grid pattern with points at :math:`(x, y)` where :math:`x` and :math:`y` are the grid indices.
    The ordering of the points can be specified as "xy" or "yx". This determines the inner and outer loop order
    when iterating over the grid points.

    * If "xy" is selected, the points are ordered with inner loop over "x" and outer loop over "y".
    * If "yx" is selected, the points are ordered with inner loop over "y" and outer loop over "x".

    For example, the grid pattern points with :math:`X = (0, 1, 2)` and :math:`Y = (3, 4)`:

    * "xy" ordering: :math:`[(0, 3), (1, 3), (2, 3), (1, 4), (2, 4), (2, 4)]`
    * "yx" ordering: :math:`[(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4)]`
    """

    shape = sensor.cfg.shape  # define in RayCasterArrayCfg

    # height scan: height = sensor_height - hit_point_z - offset
    scan = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

    # TODO: check
    if ordering == "yx":
        scan = scan.reshape(-1, shape[0], shape[1])
    elif ordering == "xy":
        scan = scan.reshape(-1, shape[1], shape[0]).transpose(1, 2)
    else:
        raise ValueError(f"Invalid ordering: {ordering}. Expected 'xy' or 'yx'.")

    return scan.unsqueeze(-1)  # add a channel dimension to the output tensor
