"""Rotation representation conversions used throughout the pipeline.

We store rotations as 3x3 matrices internally (unambiguous, easy to compose) and convert
to/from two other representations at the boundaries:
  - axis-angle: what AMASS/SMPL+H pose parameters are stored as.
  - 6D continuous (Zhou et al., 2019): what the model actually predicts/regresses, because
    axis-angle and quaternions both have discontinuities that hurt regression training near
    the wrap-around points.
"""

import torch


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula, batched. axis_angle: (..., 3) -> (..., 3, 3)."""
    shape = axis_angle.shape[:-1]
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    # Avoid divide-by-zero for near-zero rotations; direction is irrelevant when angle ~ 0.
    safe_angle = torch.where(angle > 1e-8, angle, torch.ones_like(angle))
    axis = axis_angle / safe_angle

    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zeros, -z, y], dim=-1),
        torch.stack([z, zeros, -x], dim=-1),
        torch.stack([-y, x, zeros], dim=-1),
    ], dim=-2)

    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand(*shape, 3, 3)
    sin = torch.sin(angle).unsqueeze(-1)
    cos = torch.cos(angle).unsqueeze(-1)
    R = eye + sin * K + (1 - cos) * (K @ K)

    # Zero-angle rotations collapse to identity regardless of the (arbitrary) axis above.
    is_zero = (angle.squeeze(-1) <= 1e-8).unsqueeze(-1).unsqueeze(-1)
    return torch.where(is_zero, eye, R)


def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6): first two columns of the rotation matrix, concatenated.

    Must stay the exact inverse layout of rot6d_to_matrix's a1/a2 split below — a naive
    reshape of matrix[..., :, :2] flattens row-major and interleaves the two columns instead
    of concatenating them, silently producing a different (wrong) 6D convention.
    """
    col1 = matrix[..., :, 0]
    col2 = matrix[..., :, 1]
    return torch.cat([col1, col2], dim=-1)


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt, inverting matrix_to_rot6d for model outputs.

    Needed because a model regressing 6D values won't produce an orthonormal matrix on its
    own; this reconstructs the nearest valid rotation.
    """
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def yaw_only_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Extract the yaw (rotation about the world-up Y axis) component of a rotation matrix.

    Used to canonicalize a window of motion relative to which way the head is facing, since
    absolute world yaw is arbitrary (a headset has no idea which way is "world north") but
    forward-relative-to-facing-direction is exactly what's available in a real VR app.
    """
    # Project the matrix's forward axis (column 2, i.e. local +Z) onto the world XZ plane.
    forward = matrix[..., :, 2]
    yaw = torch.atan2(forward[..., 0], forward[..., 2])
    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    row0 = torch.stack([cos, zeros, sin], dim=-1)
    row1 = torch.stack([zeros, ones, zeros], dim=-1)
    row2 = torch.stack([-sin, zeros, cos], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)
