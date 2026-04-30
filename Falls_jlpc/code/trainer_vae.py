"""
trainer_vae.py
==============
Training loop for the convolutional β-VAE anomaly detector.

  * KL annealing: β linearly ramps from 0 to its target over warmup_epochs
  * Cosine annealing with warm restarts (T_0 = 25 epochs)
  * Early stopping on val reconstruction loss of NORMAL samples ONLY
    (using fall AUROC for early stopping would leak fall information)
  * Per-session AUROC reported at every epoch as a diagnostic
"""

from __future__ import annotations
import json, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, average_precision_score

from model import ConvVAE, vae_loss


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    latent_dim: int = 64
    base_filters: int = 32
    beta_kl: float = 1.0
    warmup_epochs: int = 20
    patience: int = 30
    grad_clip: float = 1.0
    seed: int = 17


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate_vae(model, loader, device, recon_w=1.0, kl_w=0.5):
    model.eval()
    scores, labels, sessions, recons = [], [], [], []
    for batch in loader:
        x = batch["x_clean"].to(device)
        s = model.anomaly_score(x, recon_weight=recon_w,
                                kl_weight=kl_w, n_samples=4)
        x_hat, mu, logvar = model(x)
        r = ((x - x_hat) ** 2).flatten(1).mean(dim=1)
        scores.extend(s.cpu().numpy())
        recons.extend(r.cpu().numpy())
        labels.extend(batch["is_fall"].numpy())
        sessions.extend(batch["session"])

    scores = np.array(scores); labels = np.array(labels); recons = np.array(recons)

    # Replace any NaN/Inf scores with the worst finite value so AUROC
    # is still computable. If the whole batch is NaN, the model has
    # diverged — return nan AUROC and let the trainer decide what to do.
    finite = np.isfinite(scores)
    n_bad = int((~finite).sum())
    if n_bad and finite.any():
        scores[~finite] = scores[finite].max()
    elif n_bad:
        # Entire batch diverged
        return {"recon_normal": float("nan"), "auroc": float("nan"),
                "aupr": float("nan"), "per_session_auroc": {},
                "n_nan_scores": n_bad, "diverged": True}

    out = {"recon_normal": float(recons[labels == 0].mean())
           if (labels == 0).any() else float("nan"),
           "n_nan_scores": n_bad}
    if labels.sum() > 0 and labels.sum() < len(labels):
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["aupr"]  = float(average_precision_score(labels, scores))

    # Per-session AUROC (where falls exist)
    per_sess = {}
    for sk in set(sessions):
        m = np.array([s == sk for s in sessions])
        y = labels[m]
        if y.sum() > 0 and y.sum() < y.size:
            per_sess[sk] = float(roc_auc_score(y, scores[m]))
    out["per_session_auroc"] = per_sess
    return out


def train_vae(train_loader, val_loader, cfg: TrainConfig,
              out_dir: Path, device: str | None = None) -> dict:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    set_seed(cfg.seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    model = ConvVAE(in_channels=1, base=cfg.base_filters,
                    latent_dim=cfg.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] ConvVAE params={n_params/1e6:.2f}M latent={cfg.latent_dim}")

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Plain cosine annealing over the full run. Warm restarts (T_0=25) caused
    # severe loss spikes around the restart epoch (KL would suddenly grow 3×)
    # that the early-stopping criterion treated as divergence. Smooth decay
    # is more stable for this small corpus.
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs)

    history = []
    best_metric, best_epoch, no_improve = float("inf"), -1, 0

    for epoch in range(1, cfg.epochs + 1):
        beta = min(1.0, epoch / max(1, cfg.warmup_epochs)) * cfg.beta_kl
        model.train()
        t0 = time.time()
        run = {"loss": 0., "recon": 0., "kl": 0.}; n = 0
        n_skipped = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x, x_hat, mu, logvar, beta=beta)

            # Skip non-finite losses rather than committing them to weights.
            # (With clamped logvar in the model this should never trigger,
            # but the safety check is cheap and prevents epoch-1-style
            # disasters from poisoning the rest of the run.)
            if not torch.isfinite(loss):
                n_skipped += x.size(0)
                opt.zero_grad()
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            bs = x.size(0); n += bs
            run["loss"]  += loss.item()  * bs
            run["recon"] += recon.item() * bs
            run["kl"]    += kl.item()    * bs
        sched.step()

        if n_skipped:
            print(f"[ep {epoch:3d}] WARNING: skipped {n_skipped} samples "
                  f"with non-finite loss")

        for k in run: run[k] /= max(n, 1)
        run["beta"] = beta
        run["lr"]   = opt.param_groups[0]["lr"]

        val = evaluate_vae(model, val_loader, device)
        log = {"epoch": epoch, "time_s": round(time.time() - t0, 1),
               **{f"train_{k}": v for k, v in run.items()},
               **{f"val_{k}":   v for k, v in val.items()}}
        history.append(log)

        print(f"[ep {epoch:3d}] loss={run['loss']:.3f} "
              f"recon={run['recon']:.3f} kl={run['kl']:.3f} β={beta:.2f} | "
              f"val_recon_norm={val.get('recon_normal', 0):.3f} "
              f"auroc={val.get('auroc', float('nan')):.3f}")

        m = val.get("recon_normal", float("inf"))
        if m < best_metric:
            best_metric, best_epoch, no_improve = m, epoch, 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "cfg": cfg.__dict__},
                       out_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"[early-stop] no improvement for {cfg.patience} epochs")
                break

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {"best_epoch": best_epoch, "best_val_recon_normal": best_metric}
