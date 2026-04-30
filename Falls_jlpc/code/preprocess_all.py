#!/usr/bin/env python3
"""
preprocess_all.py
-------------------------
Convert every .dat in the corpus into a cached .npy spectrogram. The
cache key is derived from the PipelineConfig hash, so changing any
preprocessing parameter automatically invalidates the cache.

Run:
    python preprocess_all.py
    python preprocess_all.py --cache_dir cache --workers 4
"""

from __future__ import annotations
import argparse
from pathlib import Path


from labels import load_corpus
from preprocess import PipelineConfig
from dataset import precompute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="..")
    ap.add_argument("--cache_dir", default="cache",
                    help="Where to put precomputed .npy files. "
                         "Subdirs are created per-session.")
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    cache = Path(args.cache_dir).resolve()

    samples = load_corpus(root)
    print(f"[preprocess] {len(samples)} files to consider")

    cfg = PipelineConfig()
    precompute(samples, cfg, cache, verbose=True)


if __name__ == "__main__":
    main()
