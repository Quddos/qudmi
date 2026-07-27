"""Evaluate a trained checkpoint on held-out data: MPJPE, jitter, and per-loss breakdown.

Usage:
    python -m qudmi.train.evaluate --checkpoint checkpoints/best.pt --split test
"""

import argparse
from pathlib import Path

import torch

from qudmi.data.amass import GENDER_CODES, NUM_BODY_JOINTS
from qudmi.data.rotations import rot6d_to_matrix
from qudmi.data.smplh import forward_kinematics_from_rotmats, load_smplh_model
from qudmi.models.transformer import QudmiTransformer
from qudmi.train.losses import ROT_DIM, pose_loss

CODE_TO_GENDER = {v: k for k, v in GENDER_CODES.items()}


def reconstruct_joint_positions(pose: torch.Tensor, betas: torch.Tensor, gender_code: torch.Tensor, models: dict):
    """pose: (N, 135) = [22x6D rotation | 3D root translation]. betas: (N, 16). gender_code: (N,).

    Runs FK per gender group (each SMPL+H body model's J_regressor/shapedirs are gender-specific)
    and reassembles results back into the original (N, ...) order.
    """
    N = pose.shape[0]
    rot6d = pose[:, :ROT_DIM].reshape(N, NUM_BODY_JOINTS, 6)
    rotmats = rot6d_to_matrix(rot6d)
    root_trans = pose[:, ROT_DIM:]

    positions = torch.zeros(N, NUM_BODY_JOINTS, 3, device=pose.device)
    for code, gender in CODE_TO_GENDER.items():
        mask = gender_code == code
        if not mask.any() or gender not in models:
            continue
        pos, _ = forward_kinematics_from_rotmats(models[gender], betas[mask], rotmats[mask], root_trans[mask])
        positions[mask] = pos
    return positions


@torch.no_grad()
def evaluate(model, loader, models, device):
    """MPJPE only for now. SPEC.md also calls for jitter and foot-skate metrics, but both need
    predictions from temporally consecutive windows of the *same* sequence in order -- this
    loader draws shuffled, independent windows (matching how training data is stored), so there's
    no valid "next frame" to measure motion smoothness against here. Deferred to a sequence-level
    evaluation pass (see docs/ROADMAP.md) rather than computed incorrectly against shuffled data.
    """
    total_loss, total_rot, total_trans, n_batches = 0.0, 0.0, 0.0, 0
    total_mpjpe_mm, n_samples = 0.0, 0

    for X, Y, betas, gender_code in loader:
        X, Y = X.to(device), Y.to(device)
        betas, gender_code = betas.to(device), gender_code.to(device)

        pred = model(X)
        loss, parts = pose_loss(pred, Y)
        total_loss += loss.item()
        total_rot += parts["rot_loss"]
        total_trans += parts["trans_loss"]
        n_batches += 1

        pred_pos = reconstruct_joint_positions(pred, betas, gender_code, models)
        target_pos = reconstruct_joint_positions(Y, betas, gender_code, models)
        per_joint_error_mm = (pred_pos - target_pos).norm(dim=-1) * 1000.0  # meters -> mm
        total_mpjpe_mm += per_joint_error_mm.mean().item() * X.shape[0]
        n_samples += X.shape[0]

    return {
        "loss": total_loss / n_batches,
        "rot_loss": total_rot / n_batches,
        "trans_loss": total_trans / n_batches,
        "mpjpe_mm": total_mpjpe_mm / n_samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--body-model-dir", default="data/body_models/smplx/smplh")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data = torch.load(Path(args.data_dir) / f"{args.split}.pt")
    dataset = torch.utils.data.TensorDataset(data["X"], data["Y"], data["betas"], data["gender_code"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    body_model_dir = Path(args.body_model_dir)
    models = {}
    for gender, filename in [("female", "SMPLH_FEMALE.pkl"), ("male", "SMPLH_MALE.pkl")]:
        path = body_model_dir / filename
        if path.exists():
            models[gender] = load_smplh_model(str(path), device=args.device)

    model = QudmiTransformer().to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    metrics = evaluate(model, loader, models, args.device)
    print(f"Split: {args.split} ({len(dataset)} windows)")
    for k, v in metrics.items():
        print(f"  {k}: {v:.5f}")


if __name__ == "__main__":
    main()
