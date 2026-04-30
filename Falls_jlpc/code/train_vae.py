#!/usr/bin/env python3
"""
train_vae.py
--------------------
Train the convolutional β-VAE anomaly detector on non-fall activities.

Two split protocols supported:
  --split standard    : (session,subject)-disjoint random partition
  --split holdout     : leave one entire session out for testing
                        (use --holdout_session SESSION_KEY)

Run:
    python train_vae.py --epochs 200
    python train_vae.py --split holdout --holdout_session mar2019_cumbria
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path


from labels import load_corpus, summarize
from splits import subject_disjoint_split, session_holdout_split
from preprocess import PipelineConfig
from dataset import make_loaders
from trainer_vae import TrainConfig, train_vae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="..")
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--out_dir",   default="runs/vae_default")

    ap.add_argument("--split", choices=["standard", "holdout"],
                    default="standard")
    ap.add_argument("--holdout_session", default="mar2019_cumbria",
                    help="For --split holdout: which session to hold out.")

    ap.add_argument("--epochs",        type=int, default=200)
    ap.add_argument("--batch_size",    type=int, default=64)
    ap.add_argument("--lr",            type=float, default=1e-3)
    ap.add_argument("--weight_decay",  type=float, default=1e-4)
    ap.add_argument("--latent_dim",    type=int, default=64)
    ap.add_argument("--base_filters",  type=int, default=32)
    ap.add_argument("--beta_kl",       type=float, default=1.0)
    ap.add_argument("--warmup_epochs", type=int, default=20)
    ap.add_argument("--patience",      type=int, default=30)
    ap.add_argument("--seed",          type=int, default=17)
    ap.add_argument("--num_workers",   type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    samples = load_corpus(args.data_root)
    print(f"[corpus] {summarize(samples)}")

    if args.split == "standard":
        splits = subject_disjoint_split(samples, seed=args.seed,
                                        exclude_falls_from_train=True)
    else:
        splits = session_holdout_split(samples,
                                       holdout_session=args.holdout_session,
                                       seed=args.seed,
                                       exclude_falls_from_train=True)
    print(f"[splits] {splits.summary()}")
    with open(out_dir / "splits.json", "w") as f:
        json.dump(splits.summary(), f, indent=2)

    cfg = PipelineConfig()
    cache = Path(args.cache_dir).resolve()
    train_loader, val_loader, test_loader, global_stats = make_loaders(
        splits, cfg, cache_root=cache, batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    # Persist the standardization stats so evaluation can reproduce them.
    with open(out_dir / "global_stats.json", "w") as f:
        json.dump({"mean": global_stats[0] if global_stats else None,
                   "std":  global_stats[1] if global_stats else None,
                   "normalize_mode": cfg.normalize_mode}, f, indent=2)

    train_cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, latent_dim=args.latent_dim,
        base_filters=args.base_filters, beta_kl=args.beta_kl,
        warmup_epochs=args.warmup_epochs, patience=args.patience,
        seed=args.seed,
    )
    result = train_vae(train_loader, val_loader, train_cfg, out_dir)
    print(f"[train] best epoch: {result['best_epoch']}")


if __name__ == "__main__":
    main()
