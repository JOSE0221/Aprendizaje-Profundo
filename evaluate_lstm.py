"""Evaluate LSTM binary checkpoint: global metrics + per-subject CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from radar_cnn.dataset import RadarSequenceDataset
from radar_cnn.labels import parse_filename
from radar_cnn.model_lstm import LSTMBinaryClassifier
from radar_cnn.spectrogram import SpectrogramConfig
from radar_cnn.splits import discover_dat_files, subject_train_val_test


@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    non_blocking: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_pred = []
    all_y = []
    all_sid = []
    for x, y, sid in loader:
        x = x.to(device, non_blocking=non_blocking)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        all_pred.append(pred)
        all_y.append(y.numpy())
        all_sid.append(np.array(sid))
    return np.concatenate(all_pred), np.concatenate(all_y), np.concatenate(all_sid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--output_csv", type=str, default=None)
    ap.add_argument("--cache_dir", type=str, default=None)
    args = ap.parse_args()

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")

    spec_cfg = SpectrogramConfig(**ckpt["spec_cfg"])
    kinematic = bool(ckpt.get("kinematic", False))
    raw_mean = ckpt["mean"]
    raw_std = ckpt["std"]
    if isinstance(raw_mean, list):
        mean = np.array(raw_mean, dtype=np.float32)
        std = np.array(raw_std, dtype=np.float32)
    else:
        mean = float(raw_mean)
        std = float(raw_std)
    seed = int(ckpt.get("seed", 42))
    frac = tuple(ckpt.get("split_fractions", (0.8, 0.1, 0.1)))

    seq_len = int(ckpt.get("seq_len", 16))
    frame_reduce = str(ckpt.get("frame_reduce", "mean"))
    lstm_cfg = ckpt.get("lstm", {})
    default_in = spec_cfg.spec_height + (2 if kinematic else 0)
    model = LSTMBinaryClassifier(
        input_size=int(lstm_cfg.get("input_size", default_in)),
        hidden_size=int(lstm_cfg.get("hidden_size", 128)),
        num_layers=int(lstm_cfg.get("num_layers", 1)),
        dropout=float(lstm_cfg.get("dropout", 0.0)),
        bidirectional=bool(lstm_cfg.get("bidirectional", False)),
    )
    model.load_state_dict(ckpt["model"])

    all_files = discover_dat_files(args.data_root)
    train_p, val_p, test_p = subject_train_val_test(all_files, parse_filename, seed=seed, fractions=frac)
    if args.split == "train":
        paths = train_p
    elif args.split == "val":
        paths = val_p
    else:
        paths = test_p

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    ds = RadarSequenceDataset(
        paths,
        spec_cfg,
        seq_len=seq_len,
        frame_reduce=frame_reduce,
        binary_labels=True,
        kinematic=kinematic,
        train_mean=mean,
        train_std=std,
        cache_dir=cache_dir,
        split_name=f"lstm_{args.split}",
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = dev.type == "cuda"
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )

    device = dev
    model = model.to(device)
    pred, y_true, sids = run_eval(model, loader, device, non_blocking=pin)

    acc = accuracy_score(y_true, pred)
    macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)
    fall_f1 = f1_score(y_true, pred, average=None, labels=[1], zero_division=0)[0]
    print(
        f"Split={args.split}  n={len(y_true)}  accuracy={acc:.4f}  macro_f1={macro_f1:.4f}  fall_f1={fall_f1:.4f}"
    )
    print("\nConfusion matrix:\n", confusion_matrix(y_true, pred, labels=[0, 1]))
    print("\n", classification_report(y_true, pred, labels=[0, 1], target_names=["non_fall", "fall"], zero_division=0))

    rows = []
    for sid in np.unique(sids):
        m = sids == sid
        acc_s = accuracy_score(y_true[m], pred[m])
        mf1_s = f1_score(y_true[m], pred[m], average="macro", zero_division=0)
        ff1_s = f1_score(y_true[m], pred[m], average=None, labels=[1], zero_division=0)[0]
        rows.append(
            {
                "subject_id": int(sid),
                "n_files": int(m.sum()),
                "accuracy": acc_s,
                "macro_f1": mf1_s,
                "fall_f1": ff1_s,
            }
        )

    df = pd.DataFrame(rows).sort_values("subject_id")
    out_csv = args.output_csv or f"eval_lstm_{args.split}_per_subject.csv"
    df.to_csv(out_csv, index=False)
    print(f"Per-subject metrics saved to {out_csv}")
    print("Subject macro-F1 mean:", df["macro_f1"].mean(), "std:", df["macro_f1"].std())
    print("Subject fall-F1 mean:", df["fall_f1"].mean(), "std:", df["fall_f1"].std())


if __name__ == "__main__":
    main()
