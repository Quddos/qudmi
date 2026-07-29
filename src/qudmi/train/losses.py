"""Training losses.

Rotation regression alone is a weak signal for pose quality: a small angular error at the hip
displaces the ankle far more than the same error at the ankle does, yet an MSE over rotation
matrices scores both identically. Adding a forward-kinematics position term makes the loss care
about where the joints actually end up. DTP (Liu et al., Virtual Reality 28:116, sec 3.5) does
the same, weighting rotation 1.0 and end-effector position 0.5.

Their third term, a foot-velocity loss, is deliberately not implemented here. It compares
consecutive predicted frames, but this model predicts a single frame from a window and the
dataset stores independently shuffled windows, so there is no "previous prediction" to difference
against. Faking it against ground-truth neighbours collapses algebraically into the position loss
and would buy nothing. It needs sequence-level training to be meaningful -- noted in
docs/ROADMAP.md rather than approximated into something that only looks like the paper.
"""

import torch
import torch.nn.functional as F

from qudmi.data.rotations import rot6d_to_matrix
from qudmi.data.smplh import NUM_BODY_JOINTS, forward_kinematics_from_rotmats

ROT_DIM = NUM_BODY_JOINTS * 6
GENDER_BY_CODE = {0: "male", 1: "female", 2: "neutral"}


def _joint_positions(rotmats: torch.Tensor, betas: torch.Tensor, gender_code: torch.Tensor, models: dict):
    """Root-relative joint positions via FK, run per gender group since the body models differ.

    Root translation is deliberately zero: global placement is already covered by the translation
    term, so including it here would double-count it and let translation error swamp the pose
    signal this term exists to provide.
    """
    positions = torch.zeros(rotmats.shape[0], NUM_BODY_JOINTS, 3, device=rotmats.device)
    zero_trans = torch.zeros(rotmats.shape[0], 3, device=rotmats.device)

    for code, gender in GENDER_BY_CODE.items():
        if gender not in models:
            continue
        mask = gender_code == code
        if not mask.any():
            continue
        pos, _ = forward_kinematics_from_rotmats(
            models[gender], betas[mask], rotmats[mask], zero_trans[mask]
        )
        positions[mask] = pos

    return positions - positions[:, :1]  # root-relative


def pose_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    betas: torch.Tensor | None = None,
    gender_code: torch.Tensor | None = None,
    models: dict | None = None,
    w_rot: float = 1.0,
    w_trans: float = 1.0,
    w_pos: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """pred, target: (batch, 135) = [22 joints x 6D rotation | 3D root translation].

    Rotation error is measured after reconstructing rotation matrices (rot6d_to_matrix) rather
    than as raw MSE over the 6D numbers: two 6D vectors can differ substantially in value while
    describing nearly the same rotation, so matrix space is the more meaningful comparison.

    The FK position term is skipped when betas/gender/models aren't supplied, so callers that
    only want the cheap terms (or lack body models) still work.
    """
    pred_mat = rot6d_to_matrix(pred[..., :ROT_DIM].reshape(*pred.shape[:-1], NUM_BODY_JOINTS, 6))
    target_mat = rot6d_to_matrix(target[..., :ROT_DIM].reshape(*target.shape[:-1], NUM_BODY_JOINTS, 6))

    rot_loss = F.mse_loss(pred_mat, target_mat)
    trans_loss = F.mse_loss(pred[..., ROT_DIM:], target[..., ROT_DIM:])

    total = w_rot * rot_loss + w_trans * trans_loss
    parts = {"rot_loss": rot_loss.item(), "trans_loss": trans_loss.item(), "pos_loss": 0.0}

    if betas is not None and gender_code is not None and models:
        pred_pos = _joint_positions(pred_mat, betas, gender_code, models)
        target_pos = _joint_positions(target_mat, betas, gender_code, models)
        pos_loss = F.mse_loss(pred_pos, target_pos)
        total = total + w_pos * pos_loss
        parts["pos_loss"] = pos_loss.item()

    return total, parts
