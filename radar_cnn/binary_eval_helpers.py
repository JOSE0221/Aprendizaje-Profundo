"""Shared helpers for binary fall eval: split verification + per-subject fall counts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from radar_cnn.splits import assert_disjoint_subject_splits


def verify_and_print_subject_splits(
    train_paths: list[Path],
    val_paths: list[Path],
    test_paths: list[Path],
    parse_fn,
) -> None:
    """Assert disjoint subject IDs and print counts + sorted IDs for audit."""
    st = {parse_fn(p)[0] for p in train_paths}
    sv = {parse_fn(p)[0] for p in val_paths}
    se = {parse_fn(p)[0] for p in test_paths}
    assert_disjoint_subject_splits(train_paths, val_paths, test_paths, parse_fn)
    print(
        "Split subject audit (subject-wise; must be disjoint):\n"
        f"  train: n_subjects={len(st)}\n"
        f"  val:   n_subjects={len(sv)}\n"
        f"  test:  n_subjects={len(se)}\n"
        f"  train subject_ids (sorted): {sorted(st)}\n"
        f"  val subject_ids (sorted):   {sorted(sv)}\n"
        f"  test subject_ids (sorted):  {sorted(se)}",
        flush=True,
    )


def per_subject_binary_table(
    sids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Per-subject rows with support for falls and confusion components.

    fall_recall = fall_tp / (fall_tp + fall_fn); NaN if no true falls in that subject.
    fall_precision = fall_tp / (fall_tp + fall_fp); NaN if model never predicts fall on positives-only slice edge cases handled.
    """
    rows = []
    for sid in np.unique(sids):
        m = sids == sid
        yt = y_true[m].astype(np.int64)
        yp = y_pred[m].astype(np.int64)
        n_files = int(m.sum())
        n_true_fall = int((yt == 1).sum())
        n_true_non_fall = int((yt == 0).sum())
        fall_tp = int(((yt == 1) & (yp == 1)).sum())
        fall_fn = int(((yt == 1) & (yp == 0)).sum())
        fall_fp = int(((yt == 0) & (yp == 1)).sum())
        fall_tn = int(((yt == 0) & (yp == 0)).sum())

        denom_r = fall_tp + fall_fn
        fall_recall = fall_tp / denom_r if denom_r > 0 else float("nan")
        denom_p = fall_tp + fall_fp
        fall_precision = fall_tp / denom_p if denom_p > 0 else float("nan")

        acc_s = accuracy_score(yt, yp)
        mf1_s = f1_score(yt, yp, average="macro", zero_division=0)

        if np.unique(yt).size < 2:
            ff1_s = float("nan")
        else:
            ff1_s = float(
                f1_score(yt, yp, average=None, labels=[1], zero_division=0)[0]
            )

        rows.append(
            {
                "subject_id": int(sid),
                "n_files": n_files,
                "n_true_fall": n_true_fall,
                "n_true_non_fall": n_true_non_fall,
                "fall_tp": fall_tp,
                "fall_fn": fall_fn,
                "fall_fp": fall_fp,
                "fall_tn": fall_tn,
                "fall_recall": fall_recall,
                "fall_precision": fall_precision,
                "accuracy": acc_s,
                "macro_f1": mf1_s,
                "fall_f1": ff1_s,
            }
        )
    return pd.DataFrame(rows).sort_values("subject_id")
