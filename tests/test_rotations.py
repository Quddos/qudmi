import torch

from qudmi.data.rotations import axis_angle_to_matrix, matrix_to_rot6d, rot6d_to_matrix


def test_axis_angle_to_matrix_is_orthonormal():
    torch.manual_seed(0)
    aa = torch.randn(100, 3) * 2.0  # includes angles > pi to exercise the wrap-around
    R = axis_angle_to_matrix(aa)
    should_be_identity = R @ R.transpose(-1, -2)
    eye = torch.eye(3).expand_as(should_be_identity)
    assert torch.allclose(should_be_identity, eye, atol=1e-5)
    det = torch.linalg.det(R)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-5)


def test_axis_angle_zero_is_identity():
    R = axis_angle_to_matrix(torch.zeros(3))
    assert torch.allclose(R, torch.eye(3), atol=1e-6)


def test_rot6d_roundtrip():
    torch.manual_seed(1)
    aa = torch.randn(50, 3)
    R = axis_angle_to_matrix(aa)
    recovered = rot6d_to_matrix(matrix_to_rot6d(R))
    assert torch.allclose(R, recovered, atol=1e-4)


def test_rot6d_to_matrix_always_orthonormal():
    # Even for arbitrary (non-rotation-derived) 6D input, output must be a valid rotation --
    # this is what lets a model's raw regression output be used directly as a rotation.
    torch.manual_seed(2)
    arbitrary = torch.randn(20, 6)
    R = rot6d_to_matrix(arbitrary)
    should_be_identity = R @ R.transpose(-1, -2)
    eye = torch.eye(3).expand_as(should_be_identity)
    assert torch.allclose(should_be_identity, eye, atol=1e-5)
