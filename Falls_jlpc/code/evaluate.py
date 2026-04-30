#!/usr/bin/env python3
"""
evaluate.py
-------------------
Full evaluation of a trained VAE: AUROC/AUPR, three operating points,
per-activity reconstruction-error stats, per-session AUROC (domain-
shift diagnostic), and the demographic fairness audit.

Run:
    python evaluate.py --ckpt runs/vae_default/best.pt
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path


from labels import load_corpus
from splits import subject_disjoint_split, session_holdout_split
from preprocess import PipelineConfig
from dataset import make_loaders
from evaluator import evaluate_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="..")
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default=None,
                    help="Default: <ckpt parent>/eval/")
    ap.add_argument("--target_recall", type=float, default=0.95)
    ap.add_argument("--split", choices=["standard", "holdout"], default="standard")
    ap.add_argument("--holdout_session", default="mar2019_cumbria")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent / "eval"

    samples = load_corpus(args.data_root)
    if args.split == "standard":
        splits = subject_disjoint_split(samples, seed=args.seed)
    else:
        splits = session_holdout_split(samples,
                                       holdout_session=args.holdout_session,
                                       seed=args.seed)

    cfg = PipelineConfig()
    cache = Path(args.cache_dir).resolve()

    # Reuse the global standardization stats saved at training time.
    stats_path = Path(args.ckpt).parent / "global_stats.json"
    global_stats = None
    if stats_path.exists():
        with open(stats_path) as f:
            d = json.load(f)
        if d.get("mean") is not None:
            global_stats = (d["mean"], d["std"])
            print(f"[eval] reusing train-fold stats: mean={d['mean']:+.4f} "
                  f"std={d['std']:.4f}")

    _, val_loader, test_loader, _ = make_loaders(
        splits, cfg, cache_root=cache, batch_size=64,
        num_workers=args.num_workers, global_stats=global_stats)

    report = evaluate_full(Path(args.ckpt), val_loader, test_loader,
                           out_dir, target_recall=args.target_recall)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
