"""Time inference for a single .dat file (binary CNN checkpoint)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from radar_cnn.dataset import RadarSpectrogramDataset
from radar_cnn.model import SmallRadarCNN
from radar_cnn.spectrogram import SpectrogramConfig


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure latency for one sample: load+preprocess+forward, and forward-only (after warmup)."
    )
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dat_file", type=str, required=True, help="Path to one .dat file.")
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--warmup", type=int, default=3, help="Forward passes to discard before timing.")
    ap.add_argument("--repeats", type=int, default=20, help="Forward-only timings after warmup (mean reported).")
    args = ap.parse_args()

    dat_path = Path(args.dat_file)
    if not dat_path.is_file():
        raise SystemExit(f"File not found: {dat_path}")

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")

    if ckpt.get("model_type") != "cnn_binary":
        raise SystemExit("Checkpoint must be cnn_binary (use train_cnn_binary output).")

    spec_cfg = SpectrogramConfig(**ckpt["spec_cfg"])
    mean = float(ckpt["mean"])
    std = float(ckpt["std"])
    cnn_base = int(ckpt.get("cnn_base", 48))
    use_head_bn = bool(ckpt.get("use_head_bn", False))

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    ds = RadarSpectrogramDataset(
        [dat_path],
        spec_cfg,
        train_mean=mean,
        train_std=std,
        cache_dir=cache_dir,
        split_name="bench_one",
        binary_labels=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallRadarCNN(num_classes=2, base=cnn_base, use_head_bn=use_head_bn).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    # End-to-end: first load of this file (may build spectrogram from .dat or cache)
    t0 = time.perf_counter()
    x, y, sid = ds[0]
    x = x.unsqueeze(0).to(device)
    sync()
    t1 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
    sync()
    t2 = time.perf_counter()
    pred = int(logits.argmax(dim=1).item())

    load_preprocess_s = t1 - t0
    forward_first_s = t2 - t1

    # Forward-only: tensor already in memory; warm up then time
    x = x.to(device)
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x)
    sync()
    times = []
    with torch.no_grad():
        for _ in range(args.repeats):
            t_a = time.perf_counter()
            _ = model(x)
            sync()
            t_b = time.perf_counter()
            times.append(t_b - t_a)

    mean_fw = sum(times) / max(len(times), 1)
    std_fw = (sum((t - mean_fw) ** 2 for t in times) / max(len(times), 1)) ** 0.5

    print(f"File: {dat_path}")
    print(f"Device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"Prediction (argmax): {pred}  (label in file: {y.item()})  subject_id: {sid}")
    print()
    print("--- First sample (cache may affect load time) ---")
    print(f"  load + preprocess + to(device): {load_preprocess_s*1000:.2f} ms")
    print(f"  forward (single):                {forward_first_s*1000:.2f} ms")
    print()
    print(f"--- Forward only after {args.warmup} warmup, mean of {args.repeats} runs ---")
    print(f"  mean: {mean_fw*1000:.3f} ms  std: {std_fw*1000:.3f} ms")
    print(f"  (throughput: {1.0/mean_fw:.1f} inferences/s)")


if __name__ == "__main__":
    main()
