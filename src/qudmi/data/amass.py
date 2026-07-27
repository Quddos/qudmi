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
import re
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

# Default sparse-tracker config: VR headset + two hand controllers. This is *a* config, not
# *the* config -- process_sequence() takes tracker_joints as a parameter so the same pipeline
# can target other sparse-input setups later (e.g. adding pelvis/feet for robotics teleop or
# budget mocap) without restructuring the code, just different joint indices and a retrain.
# See docs/VISION.md.
DEFAULT_TRACKER_JOINTS = [HEAD_JOINT, LEFT_WRIST_JOINT, RIGHT_WRIST_JOINT]
FEATURES_PER_TRACKER = 3 + 6 + 3  # position + 6D rotation + velocity
INPUT_DIM = len(DEFAULT_TRACKER_JOINTS) * FEATURES_PER_TRACKER  # 36, for the default VR config
OUTPUT_DIM = NUM_BODY_JOINTS * 6 + 3  # 22 joint rotations (6D) + root translation = 135
NUM_BETAS = 16
GENDER_CODES = {"male": 0, "female": 1, "neutral": 2}  # persisted per-window so evaluation can
                                                        # run per-sample FK with the right model


@dataclass
class SequenceSample:
    subject_id: str
    X: torch.Tensor  # (num_windows, window, INPUT_DIM)
    Y: torch.Tensor  # (num_windows, OUTPUT_DIM)
    betas: torch.Tensor  # (num_windows, NUM_BETAS) -- same body shape repeated per window
    gender_code: torch.Tensor  # (num_windows,) int, see GENDER_CODES


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


_SUBJECT_PREFIX_RE = re.compile(r"^([A-Za-z]+\d+|s\d+)", re.IGNORECASE)


def subject_id_for_path(npz_path: Path) -> str:
    """Best-effort actor identity from AMASS's per-sequence folder name.

    AMASS folders are named per-recording-session, not per-actor: ACCAD, for example, has
    "Male2Walking_c3d", "Male2Running_c3d", "Male2MartialArtsKicks_c3d" etc. all for the same
    physical actor. Using the raw folder name as the split key (as an earlier version of this
    function did) lets the same actor's body shape leak into both train and val under a
    different activity folder, silently weakening the held-out-subject evaluation. This regex
    pulls out the leading "name + number" token (e.g. "Male2", "s001") shared across a given
    actor's folders; if a folder doesn't match that pattern, it falls back to the full name
    (better to slightly over-split than to silently merge unrelated recordings).
    """
    name = npz_path.parent.name
    match = _SUBJECT_PREFIX_RE.match(name)
    return match.group(1) if match else name


def process_sequence(
    npz_path: Path,
    models: dict[str, SMPLHModel],
    window: int = 40,
    stride: int = 4,
    target_fps: float = 30.0,
    device: str = "cpu",
    tracker_joints: list[int] | None = None,
) -> SequenceSample | None:
    tracker_joints = tracker_joints if tracker_joints is not None else DEFAULT_TRACKER_JOINTS
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

    num_trackers = len(tracker_joints)
    input_dim = num_trackers * FEATURES_PER_TRACKER

    tracker_positions = positions[:, tracker_joints, :]  # (T, num_trackers, 3)
    tracker_rotmats = rotmats[:, tracker_joints, :, :]  # (T, num_trackers, 3, 3)

    pos_win = tracker_positions[frame_idx]  # (M, window, num_trackers, 3)
    rot_win = tracker_rotmats[frame_idx]  # (M, window, num_trackers, 3, 3)

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

    per_tracker = torch.cat([canon_pos, rot6d, vel], dim=-1)  # (M, window, num_trackers, 12)
    X = per_tracker.reshape(len(anchors), window, input_dim)

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

    betas_per_window = betas_t.unsqueeze(0).expand(len(anchors), -1).clone()
    gender_code = torch.full((len(anchors),), GENDER_CODES[gender], dtype=torch.long)

    return SequenceSample(
        subject_id=subject_id_for_path(npz_path), X=X, Y=Y, betas=betas_per_window, gender_code=gender_code
    )


def assign_splits(subject_ids, val_frac: float = 0.1, test_frac: float = 0.1) -> dict[str, str]:
    """Deterministic train/val/test split across a full set of subjects.

    A per-subject hash-threshold split (e.g. "put this subject in test if hash(subject) % 100
    < 10") looks reasonable but silently degenerates when the subject count is small: with only
    9 actors, there's a real chance *none* of them land in the 10%-wide test bucket by chance --
    which is exactly what happened on the initial ACCAD-only run. This instead sorts all
    subjects by a hash (deterministic, but decorrelated from naming so alphabetically-similar
    actors don't cluster together) and slices explicit counts off the front, guaranteeing
    non-empty val/test buckets whenever there are enough subjects to make that meaningful.
    """
    unique_ids = sorted(set(subject_ids), key=lambda s: hashlib.md5(s.encode()).hexdigest())
    n = len(unique_ids)
    if n < 3:
        return {s: "train" for s in unique_ids}

    n_test = min(max(1, round(n * test_frac)), n - 2)
    n_val = min(max(1, round(n * val_frac)), n - n_test - 1)

    mapping = {}
    for i, subject in enumerate(unique_ids):
        if i < n_test:
            mapping[subject] = "test"
        elif i < n_test + n_val:
            mapping[subject] = "val"
        else:
            mapping[subject] = "train"
    return mapping
