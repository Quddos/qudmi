"""Training loop for Qudmi.

Usage:
    python -m qudmi.train.train --epochs 400
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from qudmi.data.smplh import load_smplh_model
from qudmi.models.transformer import QudmiTransformer
from qudmi.train.losses import pose_loss


def load_split(data_dir: Path, split: str) -> TensorDataset:
    d = torch.load(data_dir / f"{split}.pt")
    # betas/gender are needed by the FK position term; older preprocessed files predate them.
    if "betas" not in d or "gender_code" not in d:
        raise SystemExit(
            f"{data_dir / f'{split}.pt'} has no betas/gender_code -- regenerate it with "
            "scripts/preprocess_amass.py so the FK position loss can run."
        )
    return TensorDataset(d["X"], d["Y"], d["betas"], d["gender_code"])


def load_body_models(body_model_dir: Path, device: str) -> dict:
    models = {}
    for gender, filename in [("female", "SMPLH_FEMALE.pkl"), ("male", "SMPLH_MALE.pkl")]:
        path = body_model_dir / filename
        if path.exists():
            models[gender] = load_smplh_model(str(path), device=device)
    if not models:
        raise SystemExit(f"No SMPL+H body models under {body_model_dir}; see docs/DATA.md")
    return models


def run_epoch(model, loader, optimizer, device, models, w_pos, w_trans, train: bool):
    model.train(train)
    totals = {"loss": 0.0, "rot_loss": 0.0, "trans_loss": 0.0, "pos_loss": 0.0}
    n_batches = 0

    with torch.set_grad_enabled(train):
        for X, Y, betas, gender_code in loader:
            X, Y = X.to(device), Y.to(device)
            betas, gender_code = betas.to(device), gender_code.to(device)

            pred = model(X)
            loss, parts = pose_loss(pred, Y, betas, gender_code, models, w_pos=w_pos, w_trans=w_trans)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            totals["loss"] += loss.item()
            for k, v in parts.items():
                totals[k] += v
            n_batches += 1

    averaged = {k: v / n_batches for k, v in totals.items()}
    # Checkpointing and early stopping track this rather than the raw total: it covers exactly
    # what the runtime uses (joint orientations, and where those put the joints) and excludes the
    # root translation, which the Unity side discards in favour of the tracked head position.
    averaged["pose_quality"] = averaged["rot_loss"] + w_pos * averaged["pos_loss"]
    return averaged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--body-model-dir", default="data/body_models/smplx/smplh")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="runs/qudmi")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--lr-decay-epochs", type=int, default=40,
                        help="halve the learning rate every N epochs")
    parser.add_argument("--patience", type=int, default=20,
                        help="stop after this many epochs without a new best val loss")
    parser.add_argument("--w-pos", type=float, default=0.5,
                        help="weight of the FK joint-position loss (DTP uses 0.5)")
    parser.add_argument("--w-trans", type=float, default=0.1,
                        help="weight of the root-translation loss. Low by default: the Unity "
                             "runtime anchors the rig by the tracked head and ignores the "
                             "predicted root position, and this term generalizes ~10x worse "
                             "across datasets than the others, so weighting it highly would let "
                             "an output we discard dominate the gradients.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_ds, val_ds = load_split(data_dir, "train"), load_split(data_dir, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    models = load_body_models(Path(args.body_model_dir), args.device)

    print(f"train {len(train_ds)} / val {len(val_ds)} windows | device {args.device} "
          f"| body models {list(models)}")

    model = QudmiTransformer().to(args.device)
    print(f"model parameters: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_epochs, gamma=args.lr_decay)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    best_val, best_epoch, t0 = float("inf"), 0, time.time()

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, args.device, models, args.w_pos, args.w_trans, train=True)
        va = run_epoch(model, val_loader, optimizer, args.device, models, args.w_pos, args.w_trans, train=False)
        scheduler.step()

        for k in tr:
            writer.add_scalar(f"{k}/train", tr[k], epoch)
            writer.add_scalar(f"{k}/val", va[k], epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        torch.save(model.state_dict(), checkpoint_dir / "last.pt")
        marker = ""
        if va["pose_quality"] < best_val:
            best_val, best_epoch = va["pose_quality"], epoch
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
            marker = "  <- best"

        print(f"epoch {epoch:4d}/{args.epochs} | train {tr['loss']:.5f} "
              f"(rot {tr['rot_loss']:.5f} trans {tr['trans_loss']:.5f} pos {tr['pos_loss']:.5f}) | "
              f"val {va['loss']:.5f} (rot {va['rot_loss']:.5f} trans {va['trans_loss']:.5f} "
              f"pos {va['pos_loss']:.5f}) | quality {va['pose_quality']:.5f} | lr {scheduler.get_last_lr()[0]:.2e} | "
              f"{time.time() - t0:.0f}s{marker}")

        if epoch - best_epoch >= args.patience:
            print(f"\nEarly stop: no improvement for {args.patience} epochs "
                  f"(best pose-quality {best_val:.5f} at epoch {best_epoch}).")
            break

    writer.close()
    print(f"Done. Best val pose-quality {best_val:.5f} (epoch {best_epoch}). Checkpoints in {checkpoint_dir}/")


if __name__ == "__main__":
    main()
