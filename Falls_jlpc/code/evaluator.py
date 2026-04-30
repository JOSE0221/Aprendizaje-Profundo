"""
evaluator.py
============
Final evaluation. Produces a comprehensive JSON report:

  * Aggregate AUROC, AUPR on val and test
  * Three operating points: Youden's J, F1-optimal, recall-constrained
  * Per-activity reconstruction-error distribution (diagnostic)
  * Per-session AUROC (domain-shift diagnostic)
  * Per-demographic-group fall-detection metrics (fairness audit):
      - elderly (>= 60) vs younger
      - female vs male
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, confusion_matrix,
)

from labels import ACTIVITY_NAMES
from demographics import get_demographics
from model import ConvVAE


@torch.no_grad()
def collect_scores(model, loader, device):
    model.eval()
    out = {"score": [], "is_fall": [], "activity": [],
           "session": [], "subject": [], "recon": [],
           "max_velocity_mps": []}
    for batch in loader:
        x = batch["x_clean"].to(device)
        s = model.anomaly_score(x, n_samples=8).cpu().numpy()
        x_hat, mu, logvar = model(x)
        r = ((x - x_hat) ** 2).flatten(1).mean(dim=1).cpu().numpy()

        out["score"].extend(s); out["recon"].extend(r)
        out["is_fall"].extend(batch["is_fall"].numpy())
        out["activity"].extend(batch["activity"].numpy())
        out["session"].extend(batch["session"])
        out["subject"].extend(batch["subject"])
        # Cached scalar feature — same shape as the batch
        out["max_velocity_mps"].extend(batch["max_velocity_mps"].numpy())
    return {k: (np.array(v) if k not in ("session", "subject") else list(v))
            for k, v in out.items()}


def threshold_youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.argmax(tpr - fpr))
    return float(thr[i]), {"tpr": float(tpr[i]), "fpr": float(fpr[i])}


def threshold_f1(y, s):
    p, r, thr = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    i = int(np.argmax(f1))
    return float(thr[i]), {"precision": float(p[i]),
                           "recall": float(r[i]),
                           "f1": float(f1[i])}


def threshold_recall(y, s, target_recall=0.95):
    order = np.argsort(-s)
    s_sorted, y_sorted = s[order], y[order]
    n_pos = max(int(y.sum()), 1)
    cum_tp = np.cumsum(y_sorted == 1)
    recall = cum_tp / n_pos
    feasible = np.where(recall >= target_recall)[0]
    if feasible.size == 0:
        return float(s_sorted.min()), {"note": "fallback — target unreachable"}
    i = int(feasible[0])
    return float(s_sorted[i]), {"recall_at_threshold": float(recall[i])}


def metrics_at(y, s, tau):
    yhat = (s >= tau).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "tau": float(tau),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision":   float(tp / max(tp + fp, 1)),
        "recall":      float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1":          float(2 * tp / max(2 * tp + fp + fn, 1)),
        "false_alarm_rate": float(fp / max(fp + tn, 1)),
    }


def per_activity_recon(scores: dict) -> dict:
    out = {}
    for code, name in ACTIVITY_NAMES.items():
        m = (scores["activity"] == code)
        if m.sum() == 0: continue
        r = scores["recon"][m]
        out[name] = {"n": int(m.sum()),
                     "mean": float(r.mean()), "std": float(r.std()),
                     "p10": float(np.percentile(r, 10)),
                     "p50": float(np.percentile(r, 50)),
                     "p90": float(np.percentile(r, 90))}
    return out


def per_session_auroc(scores: dict) -> dict:
    out = {}
    sessions = np.array(scores["session"])
    for sk in set(scores["session"]):
        m = (sessions == sk)
        y = scores["is_fall"][m]; s = scores["score"][m]
        if y.sum() > 0 and y.sum() < y.size:
            out[sk] = {"n": int(m.sum()),
                       "n_falls": int(y.sum()),
                       "auroc": float(roc_auc_score(y, s)),
                       "aupr":  float(average_precision_score(y, s))}
    return out


def fairness_audit(scores: dict, tau: float) -> dict:
    """Per-demographic AUROC and recall at the chosen threshold."""
    audit: dict = {}
    subjects = scores["subject"]
    sessions = scores["session"]
    is_elderly = []; is_female = []
    for subj in subjects:
        sk, lid = subj.split("::")
        d = get_demographics(sk, lid)
        is_elderly.append(bool(d and d.age and d.age >= 60))
        is_female.append(bool(d and d.gender == "F"))
    is_elderly = np.array(is_elderly); is_female = np.array(is_female)

    def slice_metrics(mask, name):
        y = scores["is_fall"][mask]; s = scores["score"][mask]
        if y.sum() == 0 or y.sum() == y.size:
            return {"name": name, "n": int(mask.sum()),
                    "n_falls": int(y.sum()),
                    "note": "insufficient positives for AUROC"}
        return {"name": name, "n": int(mask.sum()),
                "n_falls": int(y.sum()),
                "auroc": float(roc_auc_score(y, s)),
                **{k: v for k, v in metrics_at(y, s, tau).items()
                   if k in ("recall", "false_alarm_rate")}}

    audit["elderly_60+"]  = slice_metrics(is_elderly,         "elderly_60+")
    audit["younger_<60"]  = slice_metrics(~is_elderly,        "younger_<60")
    audit["female"]       = slice_metrics(is_female,          "female")
    audit["male"]         = slice_metrics(~is_female,         "male")
    return audit


def svm_baseline_compare(val_s: dict, test_s: dict) -> dict:
    """Replicates the Mexico/CENAPRECE SVM paper's protocol on the SAME
    test fold: use max-Doppler velocity as the only feature with a
    threshold tuned on val. Reports val/test AUROC, AUPR, and operating
    point. Lets the report directly compare the VAE against the
    paper's F1=0.75 number under our (stricter, subject-disjoint) split.
    """
    yv, sv = val_s["is_fall"], val_s["max_velocity_mps"]
    yt, st = test_s["is_fall"], test_s["max_velocity_mps"]

    # Sanity: only proceed if there is variation in the feature
    if sv.std() < 1e-6 or st.std() < 1e-6:
        return {"note": "max-velocity feature has zero variance; "
                        "scalar features may not have been cached"}

    auroc_v = float(roc_auc_score(yv, sv))
    aupr_v  = float(average_precision_score(yv, sv))
    auroc_t = float(roc_auc_score(yt, st))
    aupr_t  = float(average_precision_score(yt, st))

    # Tune threshold on val using F1 (as in the SVM paper)
    tau, info = threshold_f1(yv, sv)
    op = metrics_at(yt, st, tau)
    return {
        "feature": "max_doppler_velocity_mps (energy-weighted P99)",
        "tau_f1_on_val":  float(tau),
        "validation":     {"auroc": auroc_v, "aupr": aupr_v, **info},
        "test_at_tau_f1": op,
        "test_aggregate": {"auroc": auroc_t, "aupr": aupr_t},
        "comparison_with_svm_paper": {
            "paper_recall":    0.72,
            "paper_precision": 0.75,
            "paper_f1":        0.75,
            "paper_protocol":  "random file-level split (likely subject-leaky)",
            "our_protocol":    "subject-disjoint, session-stratified",
            "note": "Even matching the paper's F1 under our stricter "
                    "protocol is a stronger claim than the paper's number.",
        },
    }


def evaluate_full(model_ckpt: Path, val_loader, test_loader,
                  out_dir: Path, target_recall: float = 0.95,
                  device: str | None = None) -> dict:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(model_ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = ConvVAE(in_channels=1, base=cfg["base_filters"],
                    latent_dim=cfg["latent_dim"]).to(device)
    model.load_state_dict(ckpt["model"])

    val_s  = collect_scores(model, val_loader, device)
    test_s = collect_scores(model, test_loader, device)

    auroc_val = roc_auc_score(val_s["is_fall"], val_s["score"])
    aupr_val  = average_precision_score(val_s["is_fall"], val_s["score"])

    # Tune thresholds on val
    tau_y, info_y = threshold_youden(val_s["is_fall"], val_s["score"])
    tau_f, info_f = threshold_f1(val_s["is_fall"], val_s["score"])
    tau_r, info_r = threshold_recall(val_s["is_fall"], val_s["score"], target_recall)

    # Lock thresholds, evaluate on test
    auroc_te = roc_auc_score(test_s["is_fall"], test_s["score"])
    aupr_te  = average_precision_score(test_s["is_fall"], test_s["score"])
    op = {
        "youden":             metrics_at(test_s["is_fall"], test_s["score"], tau_y),
        "f1_optimal":         metrics_at(test_s["is_fall"], test_s["score"], tau_f),
        "recall_constrained": metrics_at(test_s["is_fall"], test_s["score"], tau_r),
    }

    report = {
        "validation": {
            "auroc": float(auroc_val), "aupr": float(aupr_val),
            "tau_youden": tau_y, "tau_f1": tau_f, "tau_recall": tau_r,
            "info_youden": info_y, "info_f1": info_f, "info_recall": info_r,
        },
        "test": {
            "auroc": float(auroc_te), "aupr": float(aupr_te),
            "operating_points": op,
            "per_activity_recon_error": per_activity_recon(test_s),
            "per_session": per_session_auroc(test_s),
            "fairness_audit_at_recall_constrained":
                fairness_audit(test_s, tau_r),
        },
        "svm_baseline_max_velocity": svm_baseline_compare(val_s, test_s),
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    np.savez(out_dir / "scores.npz",
             val_score=val_s["score"], val_is_fall=val_s["is_fall"],
             val_activity=val_s["activity"], val_recon=val_s["recon"],
             val_max_velocity=val_s["max_velocity_mps"],
             te_score=test_s["score"], te_is_fall=test_s["is_fall"],
             te_activity=test_s["activity"], te_recon=test_s["recon"],
             te_max_velocity=test_s["max_velocity_mps"])
    return report
