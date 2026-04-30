"""
data_summary.py
===============
Post-preprocessing exploratory analysis. Produces:

  text outputs (printed and written to summary/summary.json):
    - corpus composition (files × activity × session × demographics)
    - max-Doppler velocity distributions per activity (the SVM-paper
      feature — sanity-check that falls really do reach higher velocities)
    - spectral-energy distributions per activity
    - session-level domain shift indicators
    - per-subject file counts and quality flags

  PNG plots (under summary/figures/):
    - 01_files_per_session_activity.png  — bar chart, files × session × act
    - 02_max_velocity_per_activity.png   — boxplot of max-Doppler velocity
    - 03_spectral_energy_per_activity.png — boxplot of spectral energy
    - 04_age_distribution_per_session.png — histogram of subject ages
    - 05_example_spectrograms.png         — 6×3 grid: 6 activities × 3 sessions
    - 06_max_velocity_roc_baseline.png    — ROC for max-velocity-only

Usage:
    python data_summary.py --data_root .. --cache_dir cache --out_dir summary

Run AFTER `make preprocess` (needs the cached spectrograms + features).
matplotlib is imported lazily so the rest of the pipeline does not
depend on it being installed.
"""

from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from labels import load_corpus, ACTIVITY_NAMES
from manifest import build_manifest, manifest_summary
from preprocess import PipelineConfig
from dataset import (cache_path, features_path, load_features, _config_hash)


def _safe_pct(x: list[float], q: float) -> float:
    return float(np.percentile(x, q)) if x else float("nan")


def collect_per_sample_stats(samples, cfg: PipelineConfig,
                             cache_root: Path) -> list[dict]:
    """Walk the corpus, joining each sample with its cached features.

    Returns a list of dicts (one per sample) with keys: session, activity,
    activity_name, is_fall, subject, max_velocity_mps, spectral_energy_db.
    Samples whose features file is missing are skipped (with a warning).
    """
    h = _config_hash(cfg)
    rows = []
    missing = 0
    for s in samples:
        feats = load_features(cache_root, s, h)
        if not feats:
            missing += 1
            continue
        rows.append({
            "session":             s.session.key,
            "activity":            s.activity,
            "activity_name":       s.label_name,
            "is_fall":             s.is_fall,
            "subject":             s.subject_id,
            "max_velocity_mps":    float(feats.get("max_doppler_velocity_mps", 0.0)),
            "spectral_energy_db":  float(feats.get("spectral_energy_db", 0.0)),
        })
    if missing:
        print(f"  [warn] {missing} samples have no cached features; "
              f"re-run `make preprocess` if you want them included")
    return rows


def text_summary(rows: list[dict], manifest_rows: list[dict]) -> dict:
    """Compute the text summary block (everything plottable too, but as
    JSON-friendly numbers)."""
    by_act = defaultdict(list)
    by_sess = defaultdict(list)
    by_act_sess = defaultdict(list)
    for r in rows:
        by_act[r["activity_name"]].append(r["max_velocity_mps"])
        by_sess[r["session"]].append(r["max_velocity_mps"])
        by_act_sess[(r["activity_name"], r["session"])].append(r["max_velocity_mps"])

    velocity_summary = {}
    for act, xs in by_act.items():
        velocity_summary[act] = {
            "n":   len(xs),
            "mean": float(np.mean(xs)) if xs else 0.0,
            "std":  float(np.std(xs))  if xs else 0.0,
            "p10": _safe_pct(xs, 10),
            "p50": _safe_pct(xs, 50),
            "p90": _safe_pct(xs, 90),
            "max": float(np.max(xs))   if xs else 0.0,
        }

    energy_summary = {}
    for act, xs in defaultdict(list, {a: [r["spectral_energy_db"] for r in rows
                                          if r["activity_name"] == a]
                                      for a in ACTIVITY_NAMES.values()}).items():
        if not xs: continue
        energy_summary[act] = {
            "n": len(xs), "mean": float(np.mean(xs)),
            "p10": _safe_pct(xs, 10), "p50": _safe_pct(xs, 50),
            "p90": _safe_pct(xs, 90),
        }

    # Falls vs non-falls separation on the max-velocity feature alone
    falls_v   = [r["max_velocity_mps"] for r in rows if r["is_fall"]]
    nofall_v  = [r["max_velocity_mps"] for r in rows if not r["is_fall"]]
    if falls_v and nofall_v:
        from sklearn.metrics import roc_auc_score
        y = np.array([r["is_fall"] for r in rows], dtype=int)
        s = np.array([r["max_velocity_mps"] for r in rows])
        max_vel_auroc = float(roc_auc_score(y, s))
    else:
        max_vel_auroc = float("nan")

    return {
        "n_samples_with_features": len(rows),
        "n_manifest_rows":         len(manifest_rows),
        "velocity_per_activity":   velocity_summary,
        "energy_per_activity":     energy_summary,
        "max_velocity_classifier_auroc_full_corpus": max_vel_auroc,
        "max_velocity_separation": {
            "fall_p10":     _safe_pct(falls_v, 10),
            "fall_p50":     _safe_pct(falls_v, 50),
            "fall_p90":     _safe_pct(falls_v, 90),
            "non_fall_p10": _safe_pct(nofall_v, 10),
            "non_fall_p50": _safe_pct(nofall_v, 50),
            "non_fall_p90": _safe_pct(nofall_v, 90),
        },
    }


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def make_plots(rows: list[dict], manifest_rows: list[dict],
               cfg: PipelineConfig, cache_root: Path,
               figures_dir: Path) -> list[Path]:
    """Generate all the figures listed at the top of this file. Returns
    the list of paths written. Imports matplotlib lazily so the rest of
    the codebase doesn't require it."""
    import matplotlib
    matplotlib.use("Agg")           # headless
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # ----- Plot 1: files × session × activity (stacked bar) -----
    sessions_order = sorted({r["session_key"] for r in manifest_rows})
    acts_order = list(ACTIVITY_NAMES.values())
    counts = np.zeros((len(sessions_order), len(acts_order)), dtype=int)
    for r in manifest_rows:
        i = sessions_order.index(r["session_key"])
        j = acts_order.index(r["activity_name"])
        counts[i, j] += 1

    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(sessions_order))
    colors = plt.cm.tab10(np.linspace(0, 1, len(acts_order)))
    for j, act in enumerate(acts_order):
        ax.bar(sessions_order, counts[:, j], bottom=bottom,
               label=act, color=colors[j], edgecolor="black", linewidth=0.3)
        bottom += counts[:, j]
    ax.set_ylabel("# files")
    ax.set_title("Files per session, stacked by activity")
    ax.legend(loc="upper right", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p = figures_dir / "01_files_per_session_activity.png"
    plt.savefig(p, dpi=120); plt.close(); written.append(p)

    # ----- Plot 2: max-velocity boxplot per activity -----
    by_act = defaultdict(list)
    for r in rows:
        by_act[r["activity_name"]].append(r["max_velocity_mps"])
    data = [by_act[a] for a in acts_order if by_act[a]]
    labels = [a for a in acts_order if by_act[a]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], plt.cm.tab10(np.linspace(0, 1, len(labels)))):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_ylabel("Max-Doppler velocity (m/s, energy-weighted P99)")
    ax.set_title("SVM-paper baseline feature distribution per activity\n"
                 "Falls should be visibly higher than other activities")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p = figures_dir / "02_max_velocity_per_activity.png"
    plt.savefig(p, dpi=120); plt.close(); written.append(p)

    # ----- Plot 3: spectral-energy boxplot per activity -----
    by_e = defaultdict(list)
    for r in rows:
        by_e[r["activity_name"]].append(r["spectral_energy_db"])
    data = [by_e[a] for a in acts_order if by_e[a]]
    labels = [a for a in acts_order if by_e[a]]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], plt.cm.tab10(np.linspace(0, 1, len(labels)))):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_ylabel("Mean spectral magnitude (dB)")
    ax.set_title("Spectral energy distribution per activity")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p = figures_dir / "03_spectral_energy_per_activity.png"
    plt.savefig(p, dpi=120); plt.close(); written.append(p)

    # ----- Plot 4: age distribution per session -----
    ages_by_sess = defaultdict(list)
    for r in manifest_rows:
        if r["age"] is not None:
            ages_by_sess[r["session_key"]].append(r["age"])
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.arange(20, 105, 5)
    for sk in sessions_order:
        ages = ages_by_sess.get(sk, [])
        if ages:
            ax.hist(ages, bins=bins, alpha=0.5, label=sk, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Subject age (years)")
    ax.set_ylabel("# files (per session)")
    ax.set_title("Demographic skew across sessions\n"
                 "Lab sessions: ~25y young; community sessions: 60–98y elderly")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p = figures_dir / "04_age_distribution_per_session.png"
    plt.savefig(p, dpi=120); plt.close(); written.append(p)

    # ----- Plot 5: example spectrogram grid (6 activities × 3 sessions) -----
    sample_sessions = ["dec2017_uog", "feb2019_nghomes", "mar2019_cumbria"]
    sample_sessions = [s for s in sample_sessions if s in sessions_order][:3]
    if sample_sessions:
        # For each (activity, session), pick one cached spectrogram
        h = _config_hash(cfg)
        chosen: dict[tuple[str, int], np.ndarray] = {}
        for r in manifest_rows:
            key = (r["session_key"], r["activity_code"])
            if key in chosen: continue
            if r["session_key"] not in sample_sessions: continue
            from labels import RadarSample
            from sessions import SESSIONS
            mock = RadarSample(path=Path(r["path"]),
                               session=SESSIONS[r["session_key"]],
                               local_id=r["local_id"], activity=r["activity_code"],
                               repetition=r["repetition"])
            cp = cache_path(cache_root, mock, h)
            if cp.exists():
                chosen[key] = np.load(cp)

        n_acts = len(ACTIVITY_NAMES)
        fig, axes = plt.subplots(n_acts, len(sample_sessions),
                                 figsize=(3.2 * len(sample_sessions), 2.0 * n_acts),
                                 squeeze=False)
        for i, act in enumerate(sorted(ACTIVITY_NAMES)):
            for j, sk in enumerate(sample_sessions):
                ax = axes[i, j]
                spec = chosen.get((sk, act))
                if spec is None:
                    ax.text(0.5, 0.5, "—", ha="center", va="center",
                            transform=ax.transAxes); ax.set_xticks([]); ax.set_yticks([])
                    continue
                ax.imshow(spec, aspect="auto", origin="lower", cmap="viridis")
                ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(ACTIVITY_NAMES[act], fontsize=9)
                if i == 0:
                    ax.set_title(sk, fontsize=9)
        fig.suptitle("Example micro-Doppler spectrograms\n"
                     "(rows = activities; columns = sessions)", y=1.0)
        plt.tight_layout()
        p = figures_dir / "05_example_spectrograms.png"
        plt.savefig(p, dpi=120); plt.close(); written.append(p)

    # ----- Plot 6: ROC of max-velocity baseline on the full corpus -----
    from sklearn.metrics import roc_curve
    y = np.array([r["is_fall"] for r in rows], dtype=int)
    s = np.array([r["max_velocity_mps"] for r in rows])
    if y.sum() > 0 and y.sum() < len(y):
        fpr, tpr, _ = roc_curve(y, s)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, s)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f"max-velocity, AUROC={auc:.3f}", lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Fall detection by max-Doppler velocity alone\n"
                     "(SVM-paper baseline, full corpus)")
        ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = figures_dir / "06_max_velocity_roc_baseline.png"
        plt.savefig(p, dpi=120); plt.close(); written.append(p)

    return written


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="..")
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--out_dir",   default="summary")
    ap.add_argument("--no_plots",  action="store_true",
                    help="Compute the JSON summary only; skip matplotlib plots.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache  = Path(args.cache_dir).resolve()
    cfg = PipelineConfig()

    print("[summary] walking corpus...")
    samples = load_corpus(args.data_root)
    manifest_rows = build_manifest(args.data_root)

    print("[summary] joining with cached features...")
    rows = collect_per_sample_stats(samples, cfg, cache)

    print("[summary] computing text summary...")
    txt = text_summary(rows, manifest_rows)
    txt["manifest_summary"] = manifest_summary(manifest_rows)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(txt, f, indent=2)

    # Console pretty-print: focus on the diagnostic that matters most —
    # whether falls are separable from non-falls by max-Doppler velocity
    # alone (the SVM-paper feature).
    print("\n=========== DATA SUMMARY ===========")
    print(f"  files with cached features: {txt['n_samples_with_features']}")
    print(f"  manifest rows: {txt['n_manifest_rows']}")
    print(f"\n  Max-Doppler velocity (m/s) per activity:")
    print(f"  {'activity':16s} {'n':>4} {'p10':>7} {'p50':>7} {'p90':>7} {'max':>7}")
    for act in ACTIVITY_NAMES.values():
        v = txt["velocity_per_activity"].get(act)
        if v is None: continue
        print(f"  {act:16s} {v['n']:4d} "
              f"{v['p10']:7.2f} {v['p50']:7.2f} {v['p90']:7.2f} {v['max']:7.2f}")
    print(f"\n  Max-velocity-only AUROC (full corpus): "
          f"{txt['max_velocity_classifier_auroc_full_corpus']:.3f}")
    print(f"  → SVM paper claims F1=0.75 with this feature; expect AUROC ≥ 0.80")
    print(f"    if our preprocessing is sound and the feature is well-extracted.")

    if not args.no_plots:
        print("\n[summary] generating plots...")
        try:
            paths = make_plots(rows, manifest_rows, cfg, cache,
                               out_dir / "figures")
            for p in paths:
                print(f"           wrote {p}")
        except ImportError:
            print("[summary] matplotlib not installed — skipping plots. "
                  "Run `pip install matplotlib` to enable.")
    print(f"\n[summary] full report: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
