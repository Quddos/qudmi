"""Loss for the v0 training loop.

v0 scope note: this is rotation + root-translation regression only. docs/SPEC.md's fuller
loss design (FK-based joint-position loss, temporal smoothness, foot-contact) needs per-window
body shape (betas) persisted through preprocessing and a differentiable FK pass during
training -- deferred until this simpler loss is confirmed to actually train (see
docs/ROADMAP.md). Shipping the smallest correct thing first, not skipping it silently.
"""

import torch
import torch.nn.functional as F

from qudmi.data.rotations import rot6d_to_matrix

ROT_DIM = 22 * 6  # 22 body joints, 6D rotation each


def pose_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """pred, target: (batch, 135) = [22 joints x 6D rotation | 3D root translation].

    Rotation error is measured after reconstructing actual rotation matrices (via
    rot6d_to_matrix) rather than as raw MSE on the 6D numbers -- two 6D vectors can differ a
    lot numerically while describing nearly the same rotation, so comparing in matrix space is
    the more physically meaningful signal.
    """
    pred_rot6d = pred[..., :ROT_DIM].reshape(*pred.shape[:-1], 22, 6)
    target_rot6d = target[..., :ROT_DIM].reshape(*target.shape[:-1], 22, 6)
    pred_mat = rot6d_to_matrix(pred_rot6d)
    target_mat = rot6d_to_matrix(target_rot6d)
    rot_loss = F.mse_loss(pred_mat, target_mat)

    trans_loss = F.mse_loss(pred[..., ROT_DIM:], target[..., ROT_DIM:])

    total = rot_loss + trans_loss
    return total, {"rot_loss": rot_loss.item(), "trans_loss": trans_loss.item()}
