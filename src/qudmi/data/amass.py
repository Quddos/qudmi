"""AMASS -> (sparse tracker signal, full-body pose) training pairs.

Pipeline per sequence file:
  1. Load raw poses/trans/betas/gender from the .npz.
  2. Resample to a fixed target frame rate (nearest-frame subsampling, not interpolation —
     see resample_indices docstring for why).
  3. Run forward kinematics (qudmi.data.smplh) to get every body joint's global position and
     orientation per frame.
  4. Slice out overlapping fixed-length windows. Each window is canonicalized relative to the
     *head's* position/yaw at the window's last frame, because that's the only reference frame
     a real headset can actually provide at inference time — absolute world position/orientation
     is not something a VR app has access to, but "relative to where my head is facing right
     now" is exactly the sparse tracker signal. See docs/SPEC.md.
  5. The model input (X) is the transformed head/left-hand/right-hand trajectory across the
     window; the target (Y) is the full-body pose (root + 21 joint rotations + root position)
     at the window's last frame, expressed in that same canonical frame.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from qudmi.data.rotations import axis_angle_to_matrix, matrix_to_rot6d, yaw_only_matrix
from qudmi.data.smplh import (
    HEAD_JOINT,
    LEFT_WRIST_JOINT,
    NUM_BODY_JOINTS,
    RIGHT_WRIST_JOINT,
    SMPLHModel,
    forward_kinematics,
)

TRACKER_JOINTS = [HEAD_JOINT, LEFT_WRIST_JOINT, RIGHT_WRIST_JOINT]
FEATURES_PER_TRACKER = 3 + 6 + 3  # position + 6D rotation + velocity
INPUT_DIM = len(TRACKER_JOINTS) * FEATURES_PER_TRACKER  # 36
OUTPUT_DIM = NUM_BODY_JOINTS * 6 + 3  # 22 joint rotations (6D) + root translation = 135


@dataclass
class SequenceSample:
    subject_id: str
    X: torch.Tensor  # (num_windows, window, INPUT_DIM)
    Y: torch.Tensor  # (num_windows, OUTPUT_DIM)


def discover_sequences(amass_root: Path) -> list[Path]:
    return sorted(Path(amass_root).rglob("*_poses.npz")) or sorted(Path(amass_root).rglob("*.npz"))


def resample_indices(num_frames: int, src_fps: float, dst_fps: float) -> np.ndarray:
    """Pick a subset of existing frame indices approximating dst_fps.

    We use nearest-frame subsampling instead of interpolating between frames: rotations can't
    be linearly interpolated correctly (that requires slerp per-joint), and for a first version
    it's simpler and equally valid to just choose real recorded frames closest to the target
    timestamps rather than fabricate blended ones.
    """
    if dst_fps >= src_fps:
        return np.arange(num_frames)
    step = max(1, round(src_fps / dst_fps))
    return np.arange(0, num_frames, step)


def _subject_id(npz_path: Path) -> str:
    return npz_path.parent.name


def process_sequence(
    npz_path: Path,
    models: dict[str, SMPLHModel],
    window: int = 40,
    stride: int = 4,
    target_fps: float = 30.0,
    device: str = "cpu",
) -> SequenceSample | None:
    data = np.load(npz_path, allow_pickle=True)
    if "poses" not in data or "trans" not in data:
        return None

    gender = str(data["gender"]) if "gender" in data else "neutral"
    gender = gender.lower()
    if gender not in models:
        raise ValueError(
            f"{npz_path}: gender '{gender}' has no loaded SMPL+H model "
            f"(available: {list(models)}). Download the matching body model, see docs/DATA.md."
        )
    model = models[gender]

    poses = data["poses"].astype(np.float32)  # (T, 156)
    trans = data["trans"].astype(np.float32)  # (T, 3)
    betas = data["betas"].astype(np.float32)[:16]  # (16,)
    src_fps = float(data["mocap_framerate"]) if "mocap_framerate" in data else 120.0

    num_frames = poses.shape[0]
    if num_frames < 2:
        return None

    idx = resample_indices(num_frames, src_fps, target_fps)
    if len(idx) < window + 1:
        return None

    body_pose_aa = torch.from_numpy(poses[idx, : NUM_BODY_JOINTS * 3].reshape(-1, NUM_BODY_JOINTS, 3)).to(device)
    root_trans = torch.from_numpy(trans[idx]).to(device)
    betas_t = torch.from_numpy(betas).to(device)

    positions, rotmats = forward_kinematics(model, betas_t, body_pose_aa, root_trans)
    T = positions.shape[0]

    anchors = np.arange(window - 1, T, stride)
    if len(anchors) == 0:
        return None
    anchors_t = torch.from_numpy(anchors).long()

    # (M, window) frame indices, ascending, for each anchor's window.
    offsets = torch.arange(window - 1, -1, -1)
    frame_idx = anchors_t[:, None] - offsets[None, :]

    tracker_positions = positions[:, TRACKER_JOINTS, :]  # (T, 3, 3)
    tracker_rotmats = rotmats[:, TRACKER_JOINTS, :, :]  # (T, 3, 3, 3)

    pos_win = tracker_positions[frame_idx]  # (M, window, 3, 3)
    rot_win = tracker_rotmats[frame_idx]  # (M, window, 3, 3, 3)

    anchor_head_pos = positions[anchors_t, HEAD_JOINT, :].clone()  # (M, 3)
    anchor_head_pos_horiz = anchor_head_pos.clone()
    anchor_head_pos_horiz[:, 1] = 0.0  # keep height absolute, only cancel horizontal position
    anchor_head_rot = rotmats[anchors_t, HEAD_JOINT, :, :]  # (M, 3, 3)
    inv_yaw = yaw_only_matrix(anchor_head_rot).transpose(-1, -2)  # (M, 3, 3), inverse == transpose

    shifted_pos = pos_win - anchor_head_pos_horiz[:, None, None, :]
    canon_pos = torch.einsum("mij,mwtj->mwti", inv_yaw, shifted_pos)  # (M, window, 3, 3)
    canon_rot = torch.einsum("mij,mwtjk->mwtik", inv_yaw, rot_win)  # (M, window, 3, 3, 3)
    rot6d = matrix_to_rot6d(canon_rot)  # (M, window, 3, 6)

    vel = torch.zeros_like(canon_pos)
    vel[:, 1:] = canon_pos[:, 1:] - canon_pos[:, :-1]
    vel[:, 0] = vel[:, 1]  # no earlier frame to diff against; backward-fill

    per_tracker = torch.cat([canon_pos, rot6d, vel], dim=-1)  # (M, window, 3, 12)
    X = per_tracker.reshape(len(anchors), window, INPUT_DIM)

    root_rotmat_anchor = rotmats[anchors_t, 0, :, :]
    canon_root_rot = torch.einsum("mij,mjk->mik", inv_yaw, root_rotmat_anchor)
    root_rot6d = matrix_to_rot6d(canon_root_rot)  # (M, 6)

    body_joint_aa_anchor = body_pose_aa[anchors_t][:, 1:NUM_BODY_JOINTS, :]  # (M, 21, 3)
    body_joint_mat = axis_angle_to_matrix(body_joint_aa_anchor)
    body_joint_6d = matrix_to_rot6d(body_joint_mat).reshape(len(anchors), -1)  # (M, 126)

    root_trans_anchor = root_trans[anchors_t]  # (M, 3)
    shifted_root_trans = root_trans_anchor - anchor_head_pos_horiz
    canon_root_trans = torch.einsum("mij,mj->mi", inv_yaw, shifted_root_trans)  # (M, 3)

    Y = torch.cat([root_rot6d, body_joint_6d, canon_root_trans], dim=-1)  # (M, 135)

    return SequenceSample(subject_id=_subject_id(npz_path), X=X, Y=Y)


def split_for_subject(subject_id: str, val_pct: int = 10, test_pct: int = 10) -> str:
    """Deterministic train/val/test split by subject, stable across runs without needing a seed."""
    bucket = int(hashlib.md5(subject_id.encode()).hexdigest(), 16) % 100
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + val_pct:
        return "val"
    return "train"
