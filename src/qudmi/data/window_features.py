"""Single-window version of the canonicalization math in amass.process_sequence.

process_sequence computes this for many overlapping windows at once (vectorized across an
"M windows" batch dimension) for preprocessing throughput. This module re-expresses the exact
same math for a *single* window -- which is what a real-time runtime (e.g. the Unity plugin)
actually needs each frame, and which is a natural, testable unit. tests/test_window_features.py
checks this against process_sequence's batched output to guard against the two implementations
drifting apart; the Unity runtime reimplements this same math in C# and is checked against it
via a generated numeric fixture (see scripts/generate_unity_parity_fixture.py) rather than
trusting the C# port to match "by construction."
"""

import torch

from qudmi.data.rotations import matrix_to_rot6d, yaw_only_matrix


def canonicalize_window(
    tracker_pos: torch.Tensor,
    tracker_rot: torch.Tensor,
    anchor_head_pos: torch.Tensor,
    anchor_head_rot: torch.Tensor,
) -> torch.Tensor:
    """
    tracker_pos: (window, num_trackers, 3) global tracker positions, Y-up
    tracker_rot: (window, num_trackers, 3, 3) global tracker rotations
    anchor_head_pos: (3,) head position at the anchor (current/last) frame of the window
    anchor_head_rot: (3, 3) head rotation at the anchor frame

    Returns: (window, num_trackers * 12) -- [position(3), 6D rotation(6), velocity(3)] per
    tracker, canonicalized relative to the head's horizontal position and yaw at the anchor
    frame (the only reference frame a real headset can actually provide -- see docs/SPEC.md).
    """
    anchor_head_pos_horiz = anchor_head_pos.clone()
    anchor_head_pos_horiz[1] = 0.0  # keep height absolute, only cancel horizontal position
    inv_yaw = yaw_only_matrix(anchor_head_rot).transpose(-1, -2)  # inverse == transpose

    shifted_pos = tracker_pos - anchor_head_pos_horiz
    canon_pos = torch.einsum("ij,wtj->wti", inv_yaw, shifted_pos)  # (window, num_trackers, 3)
    canon_rot = torch.einsum("ij,wtjk->wtik", inv_yaw, tracker_rot)  # (window, num_trackers, 3, 3)
    rot6d = matrix_to_rot6d(canon_rot)  # (window, num_trackers, 6)

    vel = torch.zeros_like(canon_pos)
    vel[1:] = canon_pos[1:] - canon_pos[:-1]
    vel[0] = vel[1]  # no earlier frame to diff against; backward-fill, matches process_sequence

    per_tracker = torch.cat([canon_pos, rot6d, vel], dim=-1)  # (window, num_trackers, 12)
    return per_tracker.reshape(tracker_pos.shape[0], -1)
