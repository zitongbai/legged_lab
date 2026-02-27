from __future__ import annotations

import torch
import torch.nn.functional as F
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import RayCaster
from isaaclab.utils import configclass
from isaaclab.envs.mdp import ManagerTermBase
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def remaining_time_fraction(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns the remaining time fraction in the episode."""
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    remaining_time = 1.0 - (env.episode_length_buf[:, None] * env.step_dt) / env.max_episode_length
    return remaining_time

class HeightScanRand(ManagerTermBase):
    
    def __init__(self, cfg: ObsTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # Pre-allocate Sobel kernels for edge detection
        self.sobel_x = torch.tensor([[-1., 0., 1.],
                                     [-2., 0., 2.],
                                     [-1., 0., 1.]], dtype=torch.float32)
        self.sobel_y = torch.tensor([[-1., -2., -1.],
                                     [ 0.,  0.,  0.],
                                     [ 1.,  2.,  1.]], dtype=torch.float32)
    
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        grid_height: int = 31,
        grid_width: int = 11,
        gaussian_std_h: float = 0.1,
        dropout_prob: float = 0.05,
        missing_value: float = 0.0,
        leg_occlusion_radius: float = 0.1,  # 10cm 半径
        leg_occlusion_prob: float = 0.7,
        enable_edge_noise: bool = True,
        edge_grad_threshold: float = 0.05,
        edge_noise_std: float = 0.03
    ) -> torch.Tensor:
        """对 scanner 输出 (N, R, 3) 做增强版随机化，返回 (N, R, 3) 高度观测.

        包含：
            1. 高斯噪声
            2. 随机 dropout
            3. 腿遮挡模型
            4. 地形边缘增强噪声

        Args:
            env: 环境实例
            cfg: 随机化配置
            asset_cfg: 用于获取脚位置的资产配置
            sensor_cfg: 用于获取扫描器的传感器配置

        Returns:
            processed_scan: (N, R, 3) 随机化后的点云 (x, y, h)
        """
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        
        # 1. 准备数据
        # 获取世界坐标系下的击中点 (N, R, 3)
        ray_hits = sensor.data.ray_hits_w
        # 计算高度/距离: height = sensor_height - hit_point_z - offset
        # 注意：这里假设用户想要的是相对高度/距离作为 z 分量
        heights = sensor.data.pos_w[:, 2].unsqueeze(1) - ray_hits[..., 2] - 0.5
        
        # 构造 scan_points (N, R, 3)，其中 z 分量替换为计算出的高度
        scan_points = torch.stack([ray_hits[..., 0], ray_hits[..., 1], heights], dim=-1)
        
        num_envs, num_rays, _ = scan_points.shape
        H, W = grid_height, grid_width
        assert H * W == num_rays, (
            f"grid_height * grid_width must equal num_rays_per_scan, "
            f"but {H} * {W} != {num_rays}"
        )

        # 提取 xyz，并 reshape 成 (N, H, W) 以便进行 2D 处理
        xs = scan_points[..., 0].view(num_envs, H, W)
        ys = scan_points[..., 1].view(num_envs, H, W)
        zs = scan_points[..., 2].view(num_envs, H, W)

        # Clone 以避免修改原始数据
        h = zs.clone()

        # 2. 应用各种噪声
        h = self._apply_gaussian_noise(h, gaussian_std_h)
        h = self._apply_dropout(h, dropout_prob, missing_value)
        h = self._apply_leg_occlusion(h, xs, ys, env, asset_cfg, leg_occlusion_radius, leg_occlusion_prob, missing_value)
        h = self._apply_edge_noise(h, enable_edge_noise, edge_grad_threshold, edge_noise_std)

        # 3. 恢复形状并返回 (N, R, 3)
        h_out = h.view(num_envs, num_rays)
        
        return h_out

    def _apply_gaussian_noise(self, h, gaussian_std_h):
        """应用高斯噪声"""
        if gaussian_std_h > 0.0:
            h = h + torch.randn_like(h) * gaussian_std_h
        return h

    def _apply_dropout(self, h, dropout_prob, missing_value):
        """应用随机 Dropout"""
        if dropout_prob > 0.0:
            drop_mask = (torch.rand_like(h) < dropout_prob)
            h = torch.where(drop_mask, torch.full_like(h, missing_value), h)
        return h

    def _apply_leg_occlusion(self, h, xs, ys, env, asset_cfg, leg_occlusion_radius, leg_occlusion_prob, missing_value):
        """应用腿部遮挡模型"""
        asset: RigidObject = env.scene[asset_cfg.name]
        foot_positions = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
        
        if foot_positions is None:
            return h
            
        num_envs = h.shape[0]
        num_feet = foot_positions.shape[1]

        # 广播形状准备计算距离
        # foot: (N, num_feet, 1, 1)
        foot_x = foot_positions[..., 0].view(num_envs, num_feet, 1, 1)
        foot_y = foot_positions[..., 1].view(num_envs, num_feet, 1, 1)
        
        # grid: (N, 1, H, W)
        xs_exp = xs.unsqueeze(1)
        ys_exp = ys.unsqueeze(1)

        # 计算距离 (N, num_feet, H, W)
        dist = torch.sqrt((xs_exp - foot_x) ** 2 + (ys_exp - foot_y) ** 2)

        # 遮挡掩码
        near_any_foot = (dist < leg_occlusion_radius).any(dim=1) # (N, H, W)
        rand_mask = torch.rand_like(h) < leg_occlusion_prob
        occlude_mask = near_any_foot & rand_mask

        return torch.where(occlude_mask, torch.full_like(h, missing_value), h)

    def _apply_edge_noise(self, h, enable_edge_noise, edge_grad_threshold, edge_noise_std):
        """应用地形边缘增强噪声 (向量化实现)"""
        if not enable_edge_noise or edge_noise_std <= 0.0:
            return h
            
        # 确保 Sobel 核在正确的设备上
        if self.sobel_x.device != h.device:
            self.sobel_x = self.sobel_x.to(h.device)
            self.sobel_y = self.sobel_y.to(h.device)

        # 准备卷积输入 (N, 1, H, W)
        h_in = h.unsqueeze(1)
        
        # 计算梯度
        grad_x = F.conv2d(h_in, self.sobel_x.view(1, 1, 3, 3), padding=1)
        grad_y = F.conv2d(h_in, self.sobel_y.view(1, 1, 3, 3), padding=1)
        grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2)).squeeze(1) # (N, H, W)
        
        # 应用噪声
        edge_mask = grad_mag > edge_grad_threshold
        if edge_mask.any():
            edge_noise = torch.randn_like(h) * edge_noise_std
            h = torch.where(edge_mask, h + edge_noise, h)
            
        return h