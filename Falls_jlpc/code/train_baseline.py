#!/usr/bin/env python3
"""
train_baseline.py
-------------------------
Train a supervised baseline (binary or multiclass CNN) for comparison
against the unsupervised anomaly detector.

  --task binary     : fall vs not-fall (the direct competitor to the VAE)
  --task multiclass : 6-way activity classification (standard published baseline)

Note: the binary baseline DOES require training falls in the train fold.
We override the autoencoder's fall-isolation and route fall recordings
into train normally for these baselines only.
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)


from labels import load_corpus
from splits import subject_disjoint_split
from preprocess import PipelineConfig
from dataset import RadarSpectrogramDataset
from baselines import BinaryCNN, MulticlassCNN
from torch.utils.data import DataLoader


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_loaders_for_supervised(splits, cfg, cache_root, batch_size, num_workers):
    """Same as the VAE loaders, but DOES include fall samples in training
    (binary baseline needs them)."""
    train_ds = RadarSpectrogramDataset(splits.train, cfg, cache_root, augment=True)
    val_ds   = RadarSpectrogramDataset(splits.val,   cfg, cache_root, augment=False)
    test_ds  = RadarSpectrogramDataset(splits.test,  cfg, cache_root, augment=False)
    pin = bool(torch.cuda.is_available())  # not on MPS/CPU
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=pin, drop_last=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin),
    )


def evaluate(model, loader, device, task):
    model.eval()
    ys, ps, ss = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x_clean"].to(device)
            logits = model(x)
            if task == "binary":
                y = batch["is_fall"].numpy()
                p = logits.softmax(dim=1)[:, 1].cpu().numpy()
                yhat = (p >= 0.5).astype(int)
                ys.extend(y); ss.extend(p); ps.extend(yhat)
            else:
                y = (batch["activity"].numpy() - 1)  # codes 1..6 → 0..5
                p = logits.argmax(dim=1).cpu().numpy()
                ys.extend(y); ps.extend(p)

    ys = np.array(ys); ps = np.array(ps)
    out = {"accuracy": float(accuracy_score(ys, ps))}
    if task == "binary":
        ss = np.array(ss)
        out["auroc"] = float(roc_auc_score(ys, ss))
        out["f1_fall"] = float(f1_score(ys, ps, pos_label=1))
        out["confusion"] = confusion_matrix(ys, ps).tolist()
    else:
        out["f1_macro"] = float(f1_score(ys, ps, average="macro"))
        out["confusion"] = confusion_matrix(ys, ps).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="..")
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--out_dir",   default="runs/baseline")
    ap.add_argument("--task", choices=["binary", "multiclass"], required=True)

    ap.add_argument("--epochs",     type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed",       type=int, default=17)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    samples = load_corpus(args.data_root)
    splits = subject_disjoint_split(samples, seed=args.seed,
                                    exclude_falls_from_train=False)
    cfg = PipelineConfig()
    cache = Path(args.cache_dir).resolve()
    train_loader, val_loader, test_loader = make_loaders_for_supervised(
        splits, cfg, cache, args.batch_size, args.num_workers)

    if args.task == "binary":
        model = BinaryCNN().to(device)
        # Class weights to handle imbalance (~10% falls)
        n_normal = sum(1 for s in splits.train if not s.is_fall)
        n_fall   = sum(1 for s in splits.train if s.is_fall)
        if n_fall > 0:
            w = torch.tensor([1.0, n_normal / n_fall], dtype=torch.float32, device=device)
        else:
            w = None
        loss_fn = nn.CrossEntropyLoss(weight=w)
    else:
        model = MulticlassCNN().to(device)
        loss_fn = nn.CrossEntropyLoss()

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    best_metric = -1; best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time(); n = 0; total_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            if args.task == "binary":
                y = batch["is_fall"].to(device)
            else:
                y = (batch["activity"] - 1).to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = x.size(0); n += bs; total_loss += loss.item() * bs
        sched.step()

        val = evaluate(model, val_loader, device, args.task)
        log = {"epoch": epoch, "train_loss": total_loss / n,
               "time_s": round(time.time() - t0, 1), **val}
        history.append(log)

        primary = val.get("auroc" if args.task == "binary" else "f1_macro", val["accuracy"])
        print(f"[ep {epoch:3d}] train_loss={total_loss/n:.3f} | "
              f"val_acc={val['accuracy']:.3f} primary={primary:.3f}")

        if primary > best_metric:
            best_metric = primary; best_epoch = epoch
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "task": args.task, "args": vars(args)},
                       out / "best.pt")

    test = evaluate(model, test_loader, device, args.task)
    with open(out / "test_metrics.json", "w") as f:
        json.dump({"best_epoch": best_epoch, **test}, f, indent=2)
    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[TEST] {test}")


if __name__ == "__main__":
    main()
