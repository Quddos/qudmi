"""Minimal SMPL+H forward kinematics: joint rotations -> global joint positions/orientations.

We only need skeleton joints (not the skinned mesh surface), so this intentionally loads and
uses a small subset of the official SMPL+H model file: v_template, shapedirs, J_regressor, and
the kinematic tree. That's enough to compute where every joint is in 3D space for a given body
shape (betas) and pose (per-joint local rotations) — which is what forward kinematics means
here. Mesh-only fields (posedirs, skinning weights, faces) are never loaded.

Joint layout (standard SMPL/SMPL+H order, indices 0-21 are the body; we ignore 22-51, the
finger joints, entirely — see docs/SPEC.md for why hand articulation is out of scope):
  0 pelvis (root), 1 L_hip, 2 R_hip, 3 spine1, 4 L_knee, 5 R_knee, 6 spine2, 7 L_ankle,
  8 R_ankle, 9 spine3, 10 L_foot, 11 R_foot, 12 neck, 13 L_collar, 14 R_collar, 15 head,
  16 L_shoulder, 17 R_shoulder, 18 L_elbow, 19 R_elbow, 20 L_wrist, 21 R_wrist
"""

import pickle
from dataclasses import dataclass

import numpy as np
import torch

from qudmi.data.rotations import axis_angle_to_matrix

NUM_BODY_JOINTS = 22
HEAD_JOINT = 15
LEFT_WRIST_JOINT = 20
RIGHT_WRIST_JOINT = 21

# AMASS/SMPL+H store poses in a Z-up world convention (inherited from motion-capture rigs like
# Vicon), but game engines and this project's canonicalization (docs/SPEC.md) assume Y-up.
# This fixed rotation (x, y, z) -> (x, z, -y) converts once, right at the FK boundary, so every
# downstream consumer of forward_kinematics() gets Y-up coordinates without needing to know
# AMASS's raw convention exists.
_ZUP_TO_YUP = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
])


@dataclass
class SMPLHModel:
    v_template: torch.Tensor   # (V, 3)
    shapedirs: torch.Tensor    # (V, 3, num_betas)
    J_regressor: torch.Tensor  # (52, V), sliced to (NUM_BODY_JOINTS, V)
    parents: list              # len NUM_BODY_JOINTS, parents[0] == -1


def load_smplh_model(pkl_path: str, device: str = "cpu") -> SMPLHModel:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")

    kintree = d["kintree_table"][0]
    parents = [int(p) if p != kintree.dtype.type(-1) else -1 for p in kintree]
    # kintree_table stores the root's "parent" as -1 wrapped to the unsigned dtype's max value.
    parents = [-1 if p == np.iinfo(kintree.dtype).max else p for p in parents]
    parents = parents[:NUM_BODY_JOINTS]

    return SMPLHModel(
        v_template=torch.as_tensor(d["v_template"], dtype=torch.float32, device=device),
        shapedirs=torch.as_tensor(np.array(d["shapedirs"]), dtype=torch.float32, device=device),
        J_regressor=torch.as_tensor(
            np.array(d["J_regressor"])[:NUM_BODY_JOINTS], dtype=torch.float32, device=device
        ),
        parents=parents,
    )


def rest_joint_locations(model: SMPLHModel, betas: torch.Tensor) -> torch.Tensor:
    """betas: (num_betas,) -> rest-pose joint locations (NUM_BODY_JOINTS, 3) for this body shape."""
    num_betas = betas.shape[-1]
    v_shaped = model.v_template + torch.einsum("vij,j->vi", model.shapedirs[:, :, :num_betas], betas)
    return model.J_regressor @ v_shaped


def forward_kinematics(
    model: SMPLHModel,
    betas: torch.Tensor,
    body_pose_axis_angle: torch.Tensor,
    root_trans: torch.Tensor,
):
    """
    betas: (num_betas,)
    body_pose_axis_angle: (T, NUM_BODY_JOINTS, 3) local joint rotations per frame
    root_trans: (T, 3) global root translation per frame

    Returns:
      global_positions: (T, NUM_BODY_JOINTS, 3)
      global_rotmats:   (T, NUM_BODY_JOINTS, 3, 3)
    """
    T = body_pose_axis_angle.shape[0]
    J_rest = rest_joint_locations(model, betas)  # (NUM_BODY_JOINTS, 3)
    local_rotmats = axis_angle_to_matrix(body_pose_axis_angle)  # (T, J, 3, 3)

    global_positions = torch.zeros(T, NUM_BODY_JOINTS, 3, device=local_rotmats.device)
    global_rotmats = torch.zeros(T, NUM_BODY_JOINTS, 3, 3, device=local_rotmats.device)

    global_positions[:, 0] = J_rest[0]
    global_rotmats[:, 0] = local_rotmats[:, 0]

    for j in range(1, NUM_BODY_JOINTS):
        parent = model.parents[j]
        offset = J_rest[j] - J_rest[parent]  # (3,)
        global_rotmats[:, j] = global_rotmats[:, parent] @ local_rotmats[:, j]
        global_positions[:, j] = global_positions[:, parent] + torch.einsum(
            "tij,j->ti", global_rotmats[:, parent], offset
        )

    global_positions = global_positions + root_trans.unsqueeze(1)

    C = _ZUP_TO_YUP.to(device=local_rotmats.device, dtype=local_rotmats.dtype)
    global_positions = global_positions @ C.T
    global_rotmats = C @ global_rotmats @ C.T
    return global_positions, global_rotmats
