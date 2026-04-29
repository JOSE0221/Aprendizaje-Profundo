"""Train LSTM baseline for binary fall vs non-fall from spectrogram sequences."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from radar_cnn.config import load_yaml, spectrogram_from_dict
from radar_cnn.dataset import (
    RadarSequenceDataset,
    compute_train_statistics_sequence,
)
from radar_cnn.labels import activity_to_binary_label, parse_filename
from radar_cnn.model_lstm import LSTMBinaryClassifier, count_parameters
from radar_cnn.plot_metrics import plot_metrics_csv
from radar_cnn.splits import (
    assert_disjoint_subject_splits,
    discover_dat_files,
    subject_train_val_test,
)

_SHUFFLE_LABELS_SEED_OFFSET = 913_337


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights_from_binary_labels(labels: np.ndarray, fall_loss_multiplier: float) -> torch.Tensor:
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    w0 = 1.0 / max(n0, 1)
    w1 = 1.0 / max(n1, 1)
    w = torch.tensor([w0, w1], dtype=torch.float32)
    w = w / w.mean()
    w[1] *= float(fall_loss_multiplier)
    return w


def _git_commit_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or None
    except Exception:
        return None


def _append_experiments_log(log_path: Path, row: dict[str, object]) -> None:
    fieldnames = [
        "timestamp_utc",
        "run_dir",
        "model_type",
        "git_commit",
        "config_path",
        "seed",
        "batch_size",
        "epochs_requested",
        "epochs_ran",
        "lr",
        "best_metric",
        "patience",
        "shuffle_labels",
        "range_bin_mode",
        "range_band",
        "stft_nperseg",
        "stft_noverlap",
        "seq_len",
        "frame_reduce",
        "hidden_size",
        "num_layers",
        "bidirectional",
        "train_files",
        "val_files",
        "test_files",
        "best_epoch",
        "best_val_macro_f1",
        "best_val_loss",
        "early_stopped",
        "metrics_csv",
        "metrics_png",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _ensure_unique_run_dir(base_output_dir: Path, seed: int, timestamp_utc: str) -> Path:
    run_markers = {"best.pt", "metrics.csv", "train_meta.json", "run_config.json"}
    if not base_output_dir.exists():
        return base_output_dir
    if not base_output_dir.is_dir():
        raise SystemExit(f"output_dir is not a directory: {base_output_dir}")
    has_run_artifacts = any((base_output_dir / m).exists() for m in run_markers)
    if not has_run_artifacts:
        return base_output_dir
    ts = timestamp_utc.replace(":", "-").replace("T", "_").replace("Z", "")
    candidate = base_output_dir / f"run_{ts}_seed{seed}"
    i = 1
    while candidate.exists():
        i += 1
        candidate = base_output_dir / f"run_{ts}_seed{seed}_{i}"
    return candidate


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    non_blocking: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    non_blocking: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_y = []
    all_p = []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
        pred = logits.argmax(dim=1)
        all_y.append(y.cpu().numpy())
        all_p.append(pred.cpu().numpy())
    return total_loss / max(n, 1), np.concatenate(all_y), np.concatenate(all_p)


def _metric_improved(best_metric: str, val_loss: float, macro_f1: float, best_val_loss: float, best_f1: float) -> bool:
    if best_metric == "val_macro_f1":
        return macro_f1 > best_f1
    return val_loss < best_val_loss


def _binary_counts(paths: list[Path]) -> dict[int, int]:
    y = np.array([activity_to_binary_label(parse_filename(p)[1]) for p in paths], dtype=np.int64)
    return {0: int((y == 0).sum()), 1: int((y == 1).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="runs/lstm_binary")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--stats_max_files", type=int, default=None)
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--shuffle_labels", action="store_true")
    ap.add_argument("--best_metric", type=str, default="val_macro_f1", choices=["val_macro_f1", "val_loss"])
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = load_yaml(args.config)
    spec_cfg = spectrogram_from_dict(cfg.get("spectrogram", {}))
    train_cfg = cfg.get("training", {})
    seq_cfg = cfg.get("sequence", {})
    lstm_cfg = cfg.get("lstm", {})
    seq_len = int(seq_cfg.get("seq_len", 16))
    frame_reduce = str(seq_cfg.get("frame_reduce", "mean"))
    kinematic = bool(seq_cfg.get("kinematic", False))
    hidden_size = int(lstm_cfg.get("hidden_size", 128))
    num_layers = int(lstm_cfg.get("num_layers", 1))
    dropout = float(lstm_cfg.get("dropout", 0.0))
    bidirectional = bool(lstm_cfg.get("bidirectional", False))
    fall_loss_multiplier = float(train_cfg.get("fall_loss_multiplier", 1.0))

    all_files = discover_dat_files(args.data_root)
    if not all_files:
        raise SystemExit(f"No .dat files under {args.data_root}")
    train_paths, val_paths, test_paths = subject_train_val_test(
        all_files,
        parse_filename,
        seed=args.seed,
        fractions=tuple(train_cfg.get("split_fractions", [0.8, 0.1, 0.1])),
    )
    assert_disjoint_subject_splits(train_paths, val_paths, test_paths, parse_filename)
    print("Split sizes:", len(train_paths), len(val_paths), len(test_paths))
    print("Train binary counts:", _binary_counts(train_paths))
    print("Val binary counts:", _binary_counts(val_paths))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    mean, std = compute_train_statistics_sequence(
        train_paths,
        spec_cfg,
        seq_len=seq_len,
        frame_reduce=frame_reduce,
        cache_dir=cache_dir,
        max_files=args.stats_max_files,
        verbose=args.verbose,
        kinematic=kinematic,
    )
    if kinematic:
        print(
            f"Sequence normalization (per feature, F={len(mean)}): mean[0,1,-2,-1]="
            f"{float(mean[0]):.4f},{float(mean[1]):.4f},{float(mean[-2]):.4f},{float(mean[-1]):.4f}"
        )
    else:
        print(f"Sequence normalization: mean={mean:.6f}, std={std:.6f}")

    y_train_true = np.array([activity_to_binary_label(parse_filename(p)[1]) for p in train_paths], dtype=np.int64)
    label_override: np.ndarray | None = None
    if args.shuffle_labels:
        rng = np.random.default_rng(args.seed + _SHUFFLE_LABELS_SEED_OFFSET)
        label_override = y_train_true[rng.permutation(len(y_train_true))]

    train_ds = RadarSequenceDataset(
        train_paths,
        spec_cfg,
        seq_len=seq_len,
        frame_reduce=frame_reduce,
        binary_labels=True,
        kinematic=kinematic,
        train_mean=mean,
        train_std=std,
        cache_dir=cache_dir,
        split_name="lstm_train",
        label_override=label_override,
    )
    val_ds = RadarSequenceDataset(
        val_paths,
        spec_cfg,
        seq_len=seq_len,
        frame_reduce=frame_reduce,
        binary_labels=True,
        kinematic=kinematic,
        train_mean=mean,
        train_std=std,
        cache_dir=cache_dir,
        split_name="lstm_val",
    )

    train_labels_for_weights = label_override if label_override is not None else y_train_true
    class_weights = class_weights_from_binary_labels(train_labels_for_weights, fall_loss_multiplier).to(device)

    input_size = spec_cfg.spec_height + (2 if kinematic else 0)
    model = LSTMBinaryClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)
    print("Parameters:", count_parameters(model))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    pin_memory = device.type == "cuda"
    persist = args.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persist,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persist,
    )

    timestamp_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_commit = _git_commit_hash()
    out_dir = _ensure_unique_run_dir(Path(args.output_dir), args.seed, timestamp_utc)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {out_dir}", flush=True)
    metrics_csv = out_dir / "metrics.csv"
    best_path = out_dir / "best.pt"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_macro_f1"])

    best_f1 = float("-inf")
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    last_epoch = 0
    early_stopped = False

    def _jsonable_stats(m: object, s: object) -> tuple[object, object]:
        if isinstance(m, np.ndarray):
            return m.tolist(), np.asarray(s, dtype=np.float32).tolist()
        return m, s

    mean_ckpt, std_ckpt = _jsonable_stats(mean, std)

    nb = device.type == "cuda" and pin_memory
    for epoch in range(args.epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, non_blocking=nb)
        val_loss, y_true, y_pred = eval_epoch(model, val_loader, criterion, device, non_blocking=nb)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        last_epoch = epoch + 1
        print(
            f"Epoch {epoch+1}/{args.epochs}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_macro_f1={macro_f1:.4f}",
            flush=True,
        )
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch + 1, tr_loss, val_loss, macro_f1])

        improved = _metric_improved(args.best_metric, val_loss, macro_f1, best_val_loss, best_f1)
        if improved:
            if args.best_metric == "val_macro_f1":
                best_f1 = macro_f1
            else:
                best_val_loss = val_loss
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_type": "lstm_binary",
                    "spec_cfg": spec_cfg.__dict__,
                    "mean": mean_ckpt,
                    "std": std_ckpt,
                    "kinematic": kinematic,
                    "config_path": str(args.config),
                    "epoch": epoch,
                    "seed": args.seed,
                    "split_fractions": tuple(train_cfg.get("split_fractions", [0.8, 0.1, 0.1])),
                    "best_metric": args.best_metric,
                    "shuffle_labels": args.shuffle_labels,
                    "seq_len": seq_len,
                    "frame_reduce": frame_reduce,
                    "binary_labels": True,
                    "fall_loss_multiplier": fall_loss_multiplier,
                    "lstm": {
                        "input_size": input_size,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "dropout": dropout,
                        "bidirectional": bidirectional,
                        "num_classes": 2,
                    },
                },
                best_path,
            )
        else:
            no_improve += 1

        if args.patience > 0 and no_improve >= args.patience:
            print(
                f"Early stopping at epoch {epoch+1} (no {args.best_metric} improvement for {args.patience} epochs).",
                flush=True,
            )
            early_stopped = True
            break

    metrics_png = out_dir / "metrics.png"
    try:
        plot_metrics_csv(metrics_csv, metrics_png)
    except Exception as e:
        print(f"Metrics plot failed: {e}", flush=True)

    run_config = {
        "timestamp_utc": timestamp_utc,
        "git_commit": git_commit,
        "output_dir": str(out_dir),
        "config_path": str(args.config),
        "cli_args": vars(args),
        "config_spectrogram": cfg.get("spectrogram", {}),
        "config_sequence": seq_cfg,
        "config_lstm": lstm_cfg,
        "config_training": train_cfg,
        "resolved": {
            "kinematic": kinematic,
            "input_size": input_size,
            "seq_len": seq_len,
            "frame_reduce": frame_reduce,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "fall_loss_multiplier": fall_loss_multiplier,
        },
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    with open(out_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp_utc": timestamp_utc,
                "git_commit": git_commit,
                "model_type": "lstm_binary",
                "kinematic": kinematic,
                "mean": mean_ckpt,
                "std": std_ckpt,
                "train_files": len(train_paths),
                "val_files": len(val_paths),
                "test_files": len(test_paths),
                "shuffle_labels": args.shuffle_labels,
                "best_metric": args.best_metric,
                "patience": args.patience,
                "epochs_ran": last_epoch,
                "early_stopped": early_stopped,
                "best_epoch": best_epoch,
                "best_val_macro_f1": None if best_f1 == float("-inf") else best_f1,
                "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
                "metrics_csv": str(metrics_csv),
                "metrics_png": str(metrics_png),
            },
            f,
            indent=2,
        )

    experiments_log = out_dir.parent / "experiments_log.csv"
    _append_experiments_log(
        experiments_log,
        {
            "timestamp_utc": timestamp_utc,
            "run_dir": str(out_dir),
            "model_type": "lstm_binary",
            "git_commit": git_commit or "",
            "config_path": str(args.config),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "epochs_ran": last_epoch,
            "lr": args.lr,
            "best_metric": args.best_metric,
            "patience": args.patience,
            "shuffle_labels": args.shuffle_labels,
            "range_bin_mode": spec_cfg.range_bin_mode,
            "range_band": str(tuple(spec_cfg.range_band)),
            "stft_nperseg": spec_cfg.stft_nperseg,
            "stft_noverlap": spec_cfg.stft_noverlap,
            "seq_len": seq_len,
            "frame_reduce": frame_reduce,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "bidirectional": bidirectional,
            "train_files": len(train_paths),
            "val_files": len(val_paths),
            "test_files": len(test_paths),
            "best_epoch": best_epoch,
            "best_val_macro_f1": "" if best_f1 == float("-inf") else f"{best_f1:.6f}",
            "best_val_loss": "" if best_val_loss == float("inf") else f"{best_val_loss:.6f}",
            "early_stopped": early_stopped,
            "metrics_csv": str(metrics_csv),
            "metrics_png": str(metrics_png),
        },
    )
    print("Saved:", best_path)
    print(f"Run config: {out_dir / 'run_config.json'}", flush=True)
    print(f"Experiments log: {experiments_log}", flush=True)


if __name__ == "__main__":
    main()
