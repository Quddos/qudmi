"""Generates a small JSON fixture used to numerically verify the Unity/C# runtime's math
against this Python pipeline's math -- see docs/SPEC.md and unity_plugin/README.md for why this
exists: AMASS/this pipeline's canonical space is right-handed Y-up (after the Z-up fix in
smplh.py), but Unity is left-handed Y-up. That handedness difference has to be converted
correctly on both the sparse-input encoding path and the predicted-pose decoding path, and a
sign error there would silently produce a garbage (but not obviously-crashing) pose. Rather
than trust a from-scratch C# reimplementation "by construction," this generates reference
values here in Python and the C# side (Tests/ParityTests.cs) asserts against them directly.

Handedness conversion used throughout (derived and verified analytically, see commit message):
  position:  unity_to_python(p) = python_to_unity(p) = (p.x, p.y, -p.z)   [self-inverse]
  rotation:  R_other = F @ R @ F,  F = diag(1, 1, -1)                    [self-inverse]
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from qudmi.data.amass import HEAD_JOINT, LEFT_WRIST_JOINT, NUM_BODY_JOINTS, RIGHT_WRIST_JOINT
from qudmi.data.rotations import matrix_to_rot6d, rot6d_to_matrix
from qudmi.data.window_features import canonicalize_window

F = np.diag([1.0, 1.0, -1.0])


def flip_z(v: np.ndarray) -> np.ndarray:
    return v * np.array([1.0, 1.0, -1.0])


def conjugate_rotation(R: np.ndarray) -> np.ndarray:
    return F @ R @ F


def build_synthetic_input(window: int):
    """Synthetic head/left-wrist/right-wrist trajectory, expressed directly in Unity's
    left-handed convention (position + xyzw quaternion), as if read straight from OpenXR.
    """
    frames = []
    for t in range(window):
        yaw_deg = 5.0 * t  # head slowly turning left
        head_pos_unity = np.array([0.0, 1.6, 0.1 * t])
        head_rot_unity = Rotation.from_euler("y", yaw_deg, degrees=True)

        # Hands roughly at shoulder height/width, bobbing slightly, rotated with the head.
        lhand_pos_unity = np.array([-0.3, 1.2 + 0.01 * t, 0.2])
        rhand_pos_unity = np.array([0.3, 1.2 - 0.01 * t, 0.2])
        hand_rot_unity = Rotation.from_euler("y", yaw_deg * 0.5, degrees=True)

        frames.append({
            "head": (head_pos_unity, head_rot_unity),
            "left_wrist": (lhand_pos_unity, hand_rot_unity),
            "right_wrist": (rhand_pos_unity, hand_rot_unity),
        })
    return frames


def unity_frame_to_python(pos_unity: np.ndarray, rot_unity: Rotation):
    pos_python = flip_z(pos_unity)
    R_python = conjugate_rotation(rot_unity.as_matrix())
    return pos_python, R_python


def main():
    window = 4
    tracker_names = ["head", "left_wrist", "right_wrist"]
    frames = build_synthetic_input(window)

    tracker_pos_py = np.zeros((window, 3, 3))
    tracker_rot_py = np.zeros((window, 3, 3, 3))
    for t, frame in enumerate(frames):
        for i, name in enumerate(tracker_names):
            pos_unity, rot_unity = frame[name]
            pos_py, R_py = unity_frame_to_python(pos_unity, rot_unity)
            tracker_pos_py[t, i] = pos_py
            tracker_rot_py[t, i] = R_py

    anchor_head_pos = tracker_pos_py[-1, 0]
    anchor_head_rot = tracker_rot_py[-1, 0]

    expected_features = canonicalize_window(
        torch.from_numpy(tracker_pos_py).float(),
        torch.from_numpy(tracker_rot_py).float(),
        torch.from_numpy(anchor_head_pos).float(),
        torch.from_numpy(anchor_head_rot).float(),
    ).numpy()

    # --- Output-decode direction: a synthetic 135-dim "model prediction" ---
    rng = np.random.default_rng(42)
    joint_rotmats_py = []
    for j in range(NUM_BODY_JOINTS):
        angle_deg = 10.0 * (j % 5)  # small, varied but deterministic per joint
        axis = ["x", "y", "z"][j % 3]
        joint_rotmats_py.append(Rotation.from_euler(axis, angle_deg, degrees=True).as_matrix())
    joint_rotmats_py = np.stack(joint_rotmats_py)  # (22, 3, 3)
    root_trans_py = np.array([0.1, 0.05, -0.2])

    joint_6d = matrix_to_rot6d(torch.from_numpy(joint_rotmats_py).float()).numpy()  # (22, 6)
    synthetic_pose_135 = np.concatenate([joint_6d.reshape(-1), root_trans_py]).astype(np.float64)

    # Reference decode: 135 -> matrices -> handedness-conjugated -> Unity quaternions.
    # (Mirrors exactly what the fixture's synthetic_pose_135 already encodes, as a round-trip
    # sanity check that this script's own encode/decode agree before C# ever sees it.)
    #
    # Joints 1-21 are parent-relative (local) rotations -- frame-invariant by construction, so
    # decoding them is just "6D -> matrix -> Unity quaternion", no further composition needed
    # (see amass.py: these come straight from the AMASS pose array, unaffected by
    # canonicalization). Joint 0 (root) is different: the model predicts it *in the
    # canonicalized (head-yaw-removed) frame*, so recovering the true world-space root pose
    # requires composing the anchor's yaw back in -- this is the step most likely to be gotten
    # wrong (it's the only place decode depends on "scene state", not just the raw prediction),
    # so it's exercised here explicitly rather than assumed correct by symmetry with the other
    # joints.
    decoded_rotmats = rot6d_to_matrix(torch.from_numpy(synthetic_pose_135[:132].reshape(22, 6)).float()).numpy()

    expected_unity_local_quats = []
    for j in range(1, NUM_BODY_JOINTS):
        R_unity = conjugate_rotation(decoded_rotmats[j])
        expected_unity_local_quats.append(Rotation.from_matrix(R_unity).as_quat().tolist())

    # Recompute the anchor's yaw matrix directly (same definition as yaw_only_matrix in
    # rotations.py): forward = anchor_head_rot's column 2, yaw = atan2(fwd.x, fwd.z).
    anchor_forward = anchor_head_rot[:, 2]
    anchor_yaw = np.arctan2(anchor_forward[0], anchor_forward[2])
    cos_y, sin_y = np.cos(anchor_yaw), np.sin(anchor_yaw)
    yaw_matrix = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])

    world_root_rot_py = yaw_matrix @ decoded_rotmats[0]
    world_root_trans_py = yaw_matrix @ synthetic_pose_135[132:] + anchor_head_pos * np.array([1.0, 0.0, 1.0])

    expected_unity_root_quat = Rotation.from_matrix(conjugate_rotation(world_root_rot_py)).as_quat().tolist()
    expected_unity_root_translation = flip_z(world_root_trans_py).tolist()

    # Unity's JsonUtility (kept as the only JSON dependency, deliberately, to avoid pulling in
    # a full JSON package for one test fixture) can only deserialize fixed-schema classes with
    # flat primitive-array fields -- no dictionaries, no nested lists. Everything below is
    # flattened accordingly; ParityTests.cs reconstructs the (window, tracker, ...) indexing
    # from these flat arrays using the same fixed stride arithmetic on both sides.
    def flat_pos_quat(tracker_key):
        pos, quat = [], []
        for frame in frames:
            p, r = frame[tracker_key]
            pos.extend(p.tolist())
            quat.extend(r.as_quat().tolist())
        return pos, quat

    head_pos, head_quat = flat_pos_quat("head")
    lhand_pos, lhand_quat = flat_pos_quat("left_wrist")
    rhand_pos, rhand_quat = flat_pos_quat("right_wrist")

    fixture = {
        "window": window,
        "headPosFlat": head_pos, "headQuatFlat": head_quat,
        "leftHandPosFlat": lhand_pos, "leftHandQuatFlat": lhand_quat,
        "rightHandPosFlat": rhand_pos, "rightHandQuatFlat": rhand_quat,
        "expectedFeaturesFlat": expected_features.reshape(-1).tolist(),  # window*36, row-major
        "syntheticPose135": synthetic_pose_135.tolist(),
        # Joints 1-21 (21 entries), local/parent-relative -- no recomposition needed.
        "expectedLocalQuatsFlat": [v for quat in expected_unity_local_quats for v in quat],  # 21*4
        # Root (joint 0), recomposed with the anchor's yaw back into true world space.
        "expectedRootQuat": expected_unity_root_quat,  # 4
        "expectedRootTranslation": expected_unity_root_translation,  # 3
    }

    out_path = Path("unity_plugin/Tests/Resources/parity_case_1.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
