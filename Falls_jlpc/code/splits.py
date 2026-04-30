"""
splits.py
=========
Subject-disjoint train/val/test partitioning using GLOBAL subject IDs
(session_key::local_id), which are unique across the whole corpus —
not local IDs, which collide across sessions.

Two-pass routing:
  PASS 1: subjects who contributed at least one fall sample go
          preferentially into val/test (we need anomalies in eval).
  PASS 2: subjects with no falls fill the remaining train/val/test
          quota. This includes the entire feb2019_nghomes session,
          which has no falls at all.

Constraint: TRAIN must contain ZERO falls. If a fall slipped through
into a train-routed subject, those fall files are dropped from the
data entirely (we never train the autoencoder on falls).

We additionally support cross-session evaluation: train on subset of
sessions, test on a held-out session. Useful for measuring domain shift
between collection sites (lab vs. NG Homes vs. West Cumbria).
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Iterable

from labels import RadarSample


@dataclass
class Split:
    train: list[RadarSample]
    val:   list[RadarSample]
    test:  list[RadarSample]

    def summary(self) -> dict:
        def stats(xs):
            n_falls = sum(1 for s in xs if s.is_fall)
            sessions = sorted({s.session.key for s in xs})
            return {
                "n":           len(xs),
                "n_falls":     n_falls,
                "n_normal":    len(xs) - n_falls,
                "n_subjects":  len({s.subject_id for s in xs}),
                "sessions":    sessions,
            }
        return {"train": stats(self.train),
                "val":   stats(self.val),
                "test":  stats(self.test)}


def assert_disjoint(splits: Split) -> None:
    tr = {s.subject_id for s in splits.train}
    va = {s.subject_id for s in splits.val}
    te = {s.subject_id for s in splits.test}
    assert tr.isdisjoint(va), f"Train/val subject leak: {tr & va}"
    assert tr.isdisjoint(te), f"Train/test subject leak: {tr & te}"
    assert va.isdisjoint(te), f"Val/test subject leak: {va & te}"
    # And: train must contain zero falls
    n_train_falls = sum(1 for s in splits.train if s.is_fall)
    assert n_train_falls == 0, \
        f"Train fold contains {n_train_falls} falls — fall isolation broken"


def subject_disjoint_split(
    samples: Iterable[RadarSample],
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    test_frac:  float = 0.15,
    seed: int = 17,
    exclude_falls_from_train: bool = True,
) -> Split:
    """Subject-disjoint partition stratified by session.

    Why session-stratified rather than 'put all fall-subjects in val/test':
    the INSHEP corpus has two no-fall sessions (NG Homes and West Cumbria)
    whose subjects are systematically different from the fall-bearing
    sessions (older, community settings vs. younger, lab settings). Routing
    all fall-subjects to val/test would force train to contain ONLY no-fall
    sessions — a maximally-adversarial domain split where the model trains
    on community-elderly signatures and is tested on lab-young signatures
    (or vice versa). Empirically this pinned val AUROC at ~0.5.

    The fix: from each session, route train_frac/val_frac/test_frac of
    SUBJECTS to train/val/test independently. Each session contributes to
    every fold, so domain is balanced across folds. Falls are dropped from
    train post-hoc when exclude_falls_from_train=True. Subject disjointness
    is preserved via global subject IDs (session_key::local_id).
    """
    eps = 1e-6
    assert abs(train_frac + val_frac + test_frac - 1.0) < eps

    samples = list(samples)
    rng = random.Random(seed)

    # Group subjects by session. We use a list (sorted) inside the dict so
    # the order is reproducible across Python runs.
    by_session: dict[str, list[str]] = {}
    for s in samples:
        bucket = by_session.setdefault(s.session.key, [])
        if s.subject_id not in bucket:
            bucket.append(s.subject_id)

    train_subj: set[str] = set()
    val_subj:   set[str] = set()
    test_subj:  set[str] = set()

    for sk in sorted(by_session.keys()):
        subj_list = sorted(by_session[sk])     # deterministic
        rng.shuffle(subj_list)
        n = len(subj_list)
        if n == 0:
            continue

        # Quotas with floor guarantees: if a session has ≥3 subjects,
        # ensure at least 1 lands in val and 1 in test.
        n_val   = round(n * val_frac)
        n_test  = round(n * test_frac)
        n_train = n - n_val - n_test
        if n >= 3:
            if n_val == 0:
                n_val = 1; n_train -= 1
            if n_test == 0:
                n_test = 1; n_train -= 1
        # If a session is too small for a 3-way split, prefer train.
        if n_train < 0:
            n_train = max(0, n - n_val - n_test)
        # Sanity: quotas must sum to n.
        assert n_train + n_val + n_test == n, \
            f"quota sum mismatch for {sk}: {n_train}+{n_val}+{n_test} != {n}"

        train_subj.update(subj_list[:n_train])
        val_subj.update(  subj_list[n_train:n_train + n_val])
        test_subj.update( subj_list[n_train + n_val:])

    train, val, test = [], [], []
    for s in samples:
        sid = s.subject_id
        if sid in train_subj:
            if exclude_falls_from_train and s.is_fall:
                continue                      # drop train falls silently
            train.append(s)
        elif sid in val_subj:
            val.append(s)
        elif sid in test_subj:
            test.append(s)

    out = Split(train=train, val=val, test=test)
    assert_disjoint(out)
    return out


def session_holdout_split(
    samples: Iterable[RadarSample],
    holdout_session: str = "mar2019_cumbria",
    val_frac_within_train: float = 0.15,
    seed: int = 17,
    exclude_falls_from_train: bool = True,
) -> Split:
    """Held-out session evaluation. Train+val on six sessions, test on
    the seventh. This is the cleanest measurement of domain shift —
    the test set has zero subject overlap AND zero environment overlap
    with training. Recommended choice: hold out the elderly West
    Cumbria session to measure generalization to older users.

    The within-pool train/val partition is stratified by session for the
    same reason subject_disjoint_split is — to keep train/val balanced
    across the diverse session populations.
    """
    samples = list(samples)
    rng = random.Random(seed)

    test = [s for s in samples if s.session.key == holdout_session]
    pool = [s for s in samples if s.session.key != holdout_session]

    # Stratify the within-pool train/val split by session.
    by_session: dict[str, list[str]] = {}
    for s in pool:
        bucket = by_session.setdefault(s.session.key, [])
        if s.subject_id not in bucket:
            bucket.append(s.subject_id)

    val_subj: set[str] = set()
    train_subj: set[str] = set()
    for sk in sorted(by_session.keys()):
        subj_list = sorted(by_session[sk])
        rng.shuffle(subj_list)
        n = len(subj_list)
        n_val = max(1, round(n * val_frac_within_train)) if n >= 2 else 0
        val_subj.update(subj_list[:n_val])
        train_subj.update(subj_list[n_val:])

    train, val = [], []
    for s in pool:
        if s.subject_id in train_subj:
            if exclude_falls_from_train and s.is_fall:
                continue
            train.append(s)
        else:
            val.append(s)

    out = Split(train=train, val=val, test=test)
    assert_disjoint(out)
    return out
