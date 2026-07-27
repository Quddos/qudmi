"""CLI: turn a downloaded AMASS directory into windowed (X, Y) training tensors.

Usage:
    python scripts/preprocess_amass.py \
        --amass-root data/amass \
        --body-model-dir data/body_models/smplx/smplh \
        --out-dir data/processed
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qudmi.data.amass import assign_splits, discover_sequences, process_sequence, subject_id_for_path
from qudmi.data.smplh import load_smplh_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amass-root", default="data/amass")
    parser.add_argument("--body-model-dir", default="data/body_models/smplx/smplh")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="only process first N files (debugging)")
    args = parser.parse_args()

    body_model_dir = Path(args.body_model_dir)
    models = {}
    for gender, filename in [("female", "SMPLH_FEMALE.pkl"), ("male", "SMPLH_MALE.pkl")]:
        path = body_model_dir / filename
        if path.exists():
            models[gender] = load_smplh_model(str(path), device=args.device)
    if not models:
        raise SystemExit(f"No SMPL+H model files found under {body_model_dir}")
    print(f"Loaded body models: {list(models)}")

    sequences = discover_sequences(args.amass_root)
    if args.limit:
        sequences = sequences[: args.limit]
    print(f"Found {len(sequences)} sequence files under {args.amass_root}")

    subject_ids = [subject_id_for_path(p) for p in sequences]
    split_map = assign_splits(subject_ids)
    n_subjects = len(set(subject_ids))
    split_subject_counts = {s: sum(1 for v in split_map.values() if v == s) for s in ("train", "val", "test")}
    print(f"{n_subjects} distinct subjects -> {split_subject_counts} (by subject, not by window)")

    buckets = {"train": [], "val": [], "test": []}
    n_ok, n_skipped, n_failed = 0, 0, 0
    t0 = time.time()

    for i, path in enumerate(sequences):
        try:
            sample = process_sequence(
                path, models, window=args.window, stride=args.stride,
                target_fps=args.target_fps, device=args.device,
            )
        except Exception as e:
            print(f"  [FAIL] {path}: {e}")
            n_failed += 1
            continue

        if sample is None:
            n_skipped += 1
            continue

        split = split_map[sample.subject_id]
        buckets[split].append((sample.X.cpu(), sample.Y.cpu(), sample.betas.cpu(), sample.gender_code.cpu()))
        n_ok += 1

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  ...{i + 1}/{len(sequences)} files ({elapsed:.1f}s elapsed)")

    print(f"\nProcessed {n_ok} sequences ok, {n_skipped} skipped (too short), {n_failed} failed.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, items in buckets.items():
        if not items:
            print(f"  {split}: 0 windows (no sequences in this split)")
            continue
        X = torch.cat([x for x, _, _, _ in items], dim=0)
        Y = torch.cat([y for _, y, _, _ in items], dim=0)
        betas = torch.cat([b for _, _, b, _ in items], dim=0)
        gender_code = torch.cat([g for _, _, _, g in items], dim=0)
        torch.save({"X": X, "Y": Y, "betas": betas, "gender_code": gender_code}, out_dir / f"{split}.pt")
        print(f"  {split}: {X.shape[0]} windows -> {out_dir / f'{split}.pt'} "
              f"(X {tuple(X.shape)}, Y {tuple(Y.shape)})")


if __name__ == "__main__":
    main()
