import numpy as np
import torch
from typing import Optional

import isaaclab.utils.math as math_utils


def vel_forward_diff(data: torch.Tensor, dt: float, method: str = "central") -> torch.Tensor:
    """Compute velocity by finite differences of the input data.

    Args:
        data (torch.Tensor): Shape (N, ...).
        dt (float): Time step duration.
        method (str): "central" (default, O(dt²)) or "forward" (O(dt)).
    """
    N = data.shape[0]
    if N < 2:
        raise RuntimeError(f"Input data has only {N} frames, cannot compute velocity.")
    vel = torch.zeros_like(data)
    if method == "central":
        if N >= 3:
            vel[1:-1] = (data[2:] - data[:-2]) / (2.0 * dt)
        vel[0] = vel[1] if N >= 3 else (data[1] - data[0]) / dt
        vel[-1] = vel[-2]
    else:  # forward
        vel[:-1] = (data[1:] - data[:-1]) / dt
        vel[-1] = vel[-2]
    return vel


def ang_vel_from_quat_diff(
    quat: torch.Tensor, dt: float, in_frame: str = "body", method: str = "central"
) -> torch.Tensor:
    """Compute angular velocity from quaternion differences (vectorized).

    Uses q_prev⁻¹ * q_next to get the relative rotation in the body frame at q_prev,
    then optionally rotates to world frame.

    Args:
        quat (torch.Tensor): Shape (N, 4), wxyz, representing world-to-body rotation.
        dt (float): Time step duration.
        in_frame (str): "body" (default) or "world".
        method (str): "central" (default, O(dt²)) or "forward" (O(dt)).
    """
    if in_frame not in ("body", "world"):
        raise ValueError(f"in_frame must be 'body' or 'world', got '{in_frame}'.")

    N = quat.shape[0]
    if N < 2:
        raise RuntimeError(f"Input quaternion has only {N} frames, cannot compute angular velocity.")

    if method == "central" and N >= 3:
        q_prev = quat[:-2]  # (N-2, 4)
        q_next = quat[2:]  # (N-2, 4)
        step = 2.0 * dt
        # diff_quat is in q_prev body frame
        diff_quat = math_utils.quat_mul(math_utils.quat_conjugate(q_prev), q_next)
        omega_mid = math_utils.axis_angle_from_quat(diff_quat) / step  # (N-2, 3)
        if in_frame == "world":
            omega_mid = math_utils.quat_apply(q_prev, omega_mid)
        # pad first and last frame
        ang_vel = torch.cat([omega_mid[:1], omega_mid, omega_mid[-1:]], dim=0)
    else:
        # forward diff (or N==2 fallback)
        q_prev = quat[:-1]  # (N-1, 4)
        q_next = quat[1:]  # (N-1, 4)
        diff_quat = math_utils.quat_mul(math_utils.quat_conjugate(q_prev), q_next)
        omega = math_utils.axis_angle_from_quat(diff_quat) / dt  # (N-1, 3)
        if in_frame == "world":
            omega = math_utils.quat_apply(q_prev, omega)
        ang_vel = torch.cat([omega, omega[-1:]], dim=0)

    return ang_vel


def quat_slerp(
    q0: torch.Tensor,
    *,
    q1: Optional[torch.Tensor] = None,
    blend: Optional[torch.Tensor] = None,
    start: Optional[np.ndarray] = None,
    end: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Interpolation between consecutive rotations (Spherical Linear Interpolation).

    Args:
        q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
        q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
        blend: Interpolation coefficient between 0 (q0) and 1 (q1). Shape is (N,) or (N, M).
        start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
            the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
        end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
            the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

    Returns:
        Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
    """
    if start is not None and end is not None:
        return quat_slerp(q0=q0[start], q1=q0[end], blend=blend)
    if q0.ndim >= 2:
        blend = blend.unsqueeze(-1)  # type: ignore
    if q0.ndim >= 3:
        blend = blend.unsqueeze(-1)  # type: ignore

    qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
    cos_half_theta = (
        q0[..., qw] * q1[..., qw] + q0[..., qx] * q1[..., qx] + q0[..., qy] * q1[..., qy] + q0[..., qz] * q1[..., qz]
    )

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()  # type: ignore
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
    ratio_b = torch.sin(blend * half_theta) / sin_half_theta

    new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
    new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
    new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
    new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

    new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
    return new_q


@torch.jit.script
def linear_interpolate(x0: torch.Tensor, x1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Linear interpolate between two tensors.

    Args:
        x0 (torch.Tensor): shape (N, M)
        x1 (torch.Tensor): shape (N, M)
        blend (torch.Tensor): shape(N, 1)
    """
    return x0 * (1 - blend) + x1 * blend


@torch.jit.script
def calc_frame_blend(time: torch.Tensor, duration: torch.Tensor, num_frames: torch.Tensor, dt: torch.Tensor):
    phase = time / duration
    phase = torch.clamp(phase, min=0.0, max=1.0)

    frame_idx0 = (phase * (num_frames - 1).float()).long()
    frame_idx1 = torch.minimum(frame_idx0 + 1, num_frames - 1)
    blend = (time - frame_idx0.float() * dt) / dt

    return frame_idx0, frame_idx1, blend
