import torch

from isaaclab.sensors import RayCasterCfg
from isaaclab.utils import configclass


@configclass
class RayCasterArrayCfg(RayCasterCfg):
    shape: tuple[int, int] = (-1, -1)

    def __post_init__(self):
        resolution = self.pattern_cfg.resolution
        size = self.pattern_cfg.size

        x = torch.arange(start=-size[0] / 2, end=size[0] / 2 + 1.0e-9, step=resolution)
        y = torch.arange(start=-size[1] / 2, end=size[1] / 2 + 1.0e-9, step=resolution)
        x_len = x.numel()
        y_len = y.numel()

        self.shape = (x_len, y_len)
