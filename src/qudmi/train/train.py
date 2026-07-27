"""Training loop for Qudmi v0.

Usage:
    python -m qudmi.train.train --data-dir data/processed --epochs 50
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from qudmi.models.transformer import QudmiTransformer
from qudmi.train.losses import pose_loss


def load_split(data_dir: Path, split: str) -> TensorDataset:
    d = torch.load(data_dir / f"{split}.pt")
    return TensorDataset(d["X"], d["Y"])


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    total_loss, total_rot, total_trans, n_batches = 0.0, 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            loss, parts = pose_loss(pred, Y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_rot += parts["rot_loss"]
            total_trans += parts["trans_loss"]
            n_batches += 1
    return total_loss / n_batches, total_rot / n_batches, total_trans / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="runs/qudmi_v0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_ds = load_split(data_dir, "train")
    val_ds = load_split(data_dir, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"train windows: {len(train_ds)}, val windows: {len(val_ds)}, device: {args.device}")

    model = QudmiTransformer().to(args.device)
    print(f"model parameters: {model.num_parameters():,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    best_val_loss = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_rot, train_trans = run_epoch(model, train_loader, optimizer, args.device, train=True)
        val_loss, val_rot, val_trans = run_epoch(model, val_loader, optimizer, args.device, train=False)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("rot_loss/train", train_rot, epoch)
        writer.add_scalar("rot_loss/val", val_rot, epoch)
        writer.add_scalar("trans_loss/train", train_trans, epoch)
        writer.add_scalar("trans_loss/val", val_trans, epoch)

        print(
            f"epoch {epoch:3d}/{args.epochs} | train {train_loss:.5f} "
            f"(rot {train_rot:.5f} trans {train_trans:.5f}) | "
            f"val {val_loss:.5f} (rot {val_rot:.5f} trans {val_trans:.5f}) | "
            f"{time.time() - t0:.1f}s elapsed"
        )

        torch.save(model.state_dict(), checkpoint_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")

    writer.close()
    print(f"Done. Best val loss: {best_val_loss:.5f}. Checkpoints in {checkpoint_dir}/")


if __name__ == "__main__":
    main()
