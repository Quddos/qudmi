"""Confirms canonicalize_window (the single-window function used to generate the Unity parity
fixture) produces identical numbers to process_sequence's batched computation, so the two
implementations of the same math can't silently drift apart.
"""

import glob
from pathlib import Path

import pytest
import torch

from qudmi.data.amass import DEFAULT_TRACKER_JOINTS, HEAD_JOINT, NUM_BODY_JOINTS
from qudmi.data.smplh import forward_kinematics, load_smplh_model
from qudmi.data.window_features import canonicalize_window

ACCAD_FILES = sorted(glob.glob("data/amass/ACCAD/**/*_poses.npz", recursive=True))
BODY_MODEL = "data/body_models/smplx/smplh/SMPLH_FEMALE.pkl"


@pytest.mark.skipif(not ACCAD_FILES, reason="requires downloaded AMASS data, see docs/DATA.md")
def test_single_window_matches_batched_process_sequence():
    import numpy as np

    from qudmi.data.amass import process_sequence

    model = load_smplh_model(BODY_MODEL)
    path = Path(ACCAD_FILES[0])

    window, stride = 40, 4
    sample = process_sequence(path, {"female": model, "male": model}, window=window, stride=stride)
    assert sample is not None

    # Recompute frame 0's window (the first anchor) independently via the raw data, then compare
    # against canonicalize_window's output for that same window -- and against sample.X[0], which
    # is process_sequence's batched result for that exact window.
    data = np.load(path, allow_pickle=True)
    from qudmi.data.amass import resample_indices

    poses = data["poses"].astype(np.float32)
    trans = data["trans"].astype(np.float32)
    betas = torch.from_numpy(data["betas"].astype(np.float32)[:16])
    src_fps = float(data["mocap_framerate"])
    idx = resample_indices(poses.shape[0], src_fps, 30.0)

    body_pose_aa = torch.from_numpy(poses[idx, : NUM_BODY_JOINTS * 3].reshape(-1, NUM_BODY_JOINTS, 3))
    root_trans = torch.from_numpy(trans[idx])
    positions, rotmats = forward_kinematics(model, betas, body_pose_aa, root_trans)

    anchor = window - 1
    frame_slice = slice(anchor - window + 1, anchor + 1)
    tracker_pos = positions[frame_slice][:, DEFAULT_TRACKER_JOINTS, :]
    tracker_rot = rotmats[frame_slice][:, DEFAULT_TRACKER_JOINTS, :, :]
    anchor_head_pos = positions[anchor, HEAD_JOINT, :]
    anchor_head_rot = rotmats[anchor, HEAD_JOINT, :, :]

    independent = canonicalize_window(tracker_pos, tracker_rot, anchor_head_pos, anchor_head_rot)

    assert torch.allclose(independent, sample.X[0], atol=1e-4)
