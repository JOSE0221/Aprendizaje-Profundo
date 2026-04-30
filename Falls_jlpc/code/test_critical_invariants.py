"""
test_critical_invariants.py
===========================
Pytest suite for the things that absolutely must not regress:

  1. Filename parser handles all known naming variations
  2. Subject IDs are GLOBAL (session+local) — same local_id across
     sessions produces DIFFERENT global IDs
  3. Subject-disjoint split enforces no leakage
  4. Train fold contains zero falls when fall isolation is enabled
  5. NG Homes session is correctly flagged as has_falls=False
  6. Demographics for known collisions are different (e.g. dec2017 P36
     vs nghomes P36 are different people)

Run:
    pytest -v
"""

from __future__ import annotations
from pathlib import Path
from unittest.mock import patch


import pytest

from sessions import SESSIONS, global_subject_id
from labels import (
    parse_dat_filename, _PATTERN, ACTIVITY_NAMES, RadarSample, _canonical_local_id,
)
from demographics import get_demographics
from splits import subject_disjoint_split, assert_disjoint, Split


# ----------------------------------------------------------------------
# Filename parsing
# ----------------------------------------------------------------------
def test_canonical_id():
    assert _canonical_local_id("36") == "P36"
    assert _canonical_local_id("P36") == "P36"
    assert _canonical_local_id("P036") == "P36"
    assert _canonical_local_id("p3") == "P03"
    assert _canonical_local_id("P003") == "P03"


@pytest.mark.parametrize("name,expected", [
    ("1P36A01R1.dat",   (1, "P36", 1)),
    ("1P36A01R01.dat",  (1, "P36", 1)),
    ("6P12A06R3.dat",   (6, "P12", 3)),
    ("1_P36_A01_R1.dat", (1, "P36", 1)),
    ("1P36 A01 R1.dat", (1, "P36", 1)),
    ("6p08a06r2.dat",   (6, "P08", 2)),
    ("3P03A03R1.dat",   (3, "P03", 1)),
])
def test_filename_pattern(name, expected):
    sess = SESSIONS["dec2017_uog"]
    s = parse_dat_filename(name, sess)
    assert s is not None, f"Failed to parse {name}"
    assert s.activity == expected[0]
    assert s.local_id == expected[1]
    assert s.repetition == expected[2]


def test_filename_pattern_rejects_garbage():
    sess = SESSIONS["dec2017_uog"]
    assert parse_dat_filename("readme.txt", sess) is None
    assert parse_dat_filename("garbage.dat", sess) is None
    assert parse_dat_filename("9P36A01R1.dat", sess) is None  # K=9 invalid


# ----------------------------------------------------------------------
# Global subject IDs — the critical fix
# ----------------------------------------------------------------------
def test_global_subject_ids_distinguish_sessions():
    """P36 in dec2017 must NOT equal P36 in nghomes."""
    g1 = global_subject_id("dec2017_uog", "P36")
    g2 = global_subject_id("feb2019_nghomes", "P36")
    assert g1 != g2
    assert g1 == "dec2017_uog::P36"
    assert g2 == "feb2019_nghomes::P36"


def test_demographics_correctly_separated_for_collisions():
    """The Readme makes clear that P36 in two different sessions is
    different people. Our demographics table must reflect that."""
    d_dec = get_demographics("dec2017_uog", "P36")
    d_ng  = get_demographics("feb2019_nghomes", "P36")
    assert d_dec is not None and d_ng is not None
    # Dec 2017 P36: 27y male; NG Homes P36: 70y female
    assert d_dec.age == 27 and d_dec.gender == "M"
    assert d_ng.age == 70 and d_ng.gender == "F"


def test_p37_collision_dec2017_vs_west_cumbria():
    """Dec 2017 P37: 27y male; WC P37: 75y female."""
    a = get_demographics("dec2017_uog", "P37")
    b = get_demographics("mar2019_cumbria", "P37")
    assert a.age == 27 and a.gender == "M"
    assert b.age == 75 and b.gender == "F"


# ----------------------------------------------------------------------
# Session metadata
# ----------------------------------------------------------------------
def test_nghomes_has_no_falls_flag():
    assert SESSIONS["feb2019_nghomes"].has_falls is False


def test_cumbria_has_no_falls_flag():
    """West Cumbria — Age UK elderly community centre — 5 activities only,
    no falls (confirmed empirically: 0 fall files in 289 recordings)."""
    assert SESSIONS["mar2019_cumbria"].has_falls is False


def test_other_sessions_have_falls():
    for k in ["dec2017_uog", "mar2017_uog", "jun2017_uog", "jul2018_uog",
              "feb2019_uog"]:
        assert SESSIONS[k].has_falls is True


def test_session_folder_names_match_screenshot():
    """Folder names must exactly match what the user has on disk —
    spaces and leading digits included."""
    expected = {
        "dec2017_uog":      "1 December 2017 Dataset",
        "mar2017_uog":      "2 March 2017 Dataset",
        "jun2017_uog":      "3 June 2017 Dataset",
        "jul2018_uog":      "4 July 2018 Dataset",
        "feb2019_uog":      "5 February 2019 UoG Dataset",
        "feb2019_nghomes":  "6 February 2019 NG Homes Dataset",
        "mar2019_cumbria":  "7 March 2019 West Cumbria Dataset",
    }
    for k, folder in expected.items():
        assert SESSIONS[k].folder == folder


# ----------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------
def _mock_sample(session_key: str, local_id: str, activity: int, rep: int = 1):
    return RadarSample(
        path=Path(f"/tmp/{session_key}_{local_id}_A{activity:02d}_R{rep}.dat"),
        session=SESSIONS[session_key],
        local_id=local_id, activity=activity, repetition=rep,
    )


def test_split_zero_train_falls():
    """If exclude_falls_from_train=True, zero falls in train fold."""
    samples = []
    # A few subjects with falls
    for subj in ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08"]:
        for act in [1, 2, 3, 4, 5, 6]:
            samples.append(_mock_sample("dec2017_uog", subj, act))
    splits = subject_disjoint_split(samples, seed=0,
                                    exclude_falls_from_train=True)
    train_falls = [s for s in splits.train if s.is_fall]
    assert len(train_falls) == 0, f"Found {len(train_falls)} falls in train"


def test_split_subject_disjointness():
    samples = []
    for subj in ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08",
                 "P09", "P10"]:
        for act in [1, 2, 3, 4, 5, 6]:
            samples.append(_mock_sample("dec2017_uog", subj, act))
    splits = subject_disjoint_split(samples, seed=42)
    assert_disjoint(splits)  # Should not raise


def test_split_handles_session_with_no_falls():
    """NG Homes-only data should still split cleanly."""
    samples = []
    for subj in ["P18", "P19", "P20", "P21", "P22", "P23", "P24", "P25"]:
        for act in [1, 2, 3, 4, 5]:  # no A6 in NG Homes
            samples.append(_mock_sample("feb2019_nghomes", subj, act))
    splits = subject_disjoint_split(samples, seed=0)
    assert_disjoint(splits)
    # No falls anywhere
    for fold in [splits.train, splits.val, splits.test]:
        assert not any(s.is_fall for s in fold)


def test_split_global_ids_prevent_cross_session_leak():
    """A subject local_id 'P08' present in two different sessions must
    be treated as TWO different subjects and CAN end up in different
    folds — but no LEAKAGE across folds for the SAME global id."""
    samples = []
    # feb2019_uog::P08 contributes 6 activities incl. fall
    for act in [1, 2, 3, 4, 5, 6]:
        samples.append(_mock_sample("feb2019_uog", "P08", act))
    # feb2019_nghomes::P08 contributes 5 activities (no fall in nghomes)
    for act in [1, 2, 3, 4, 5]:
        samples.append(_mock_sample("feb2019_nghomes", "P08", act))
    # And a few other subjects
    for subj in ["P01", "P02", "P03", "P04"]:
        for act in [1, 2, 3, 4, 5, 6]:
            samples.append(_mock_sample("dec2017_uog", subj, act))

    splits = subject_disjoint_split(samples, seed=7)
    assert_disjoint(splits)
    # Both versions of P08 should appear somewhere
    all_globals = set()
    for fold in [splits.train, splits.val, splits.test]:
        for s in fold:
            all_globals.add(s.subject_id)
    assert "feb2019_uog::P08" in all_globals
    assert "feb2019_nghomes::P08" in all_globals


def test_split_stratifies_across_sessions():
    """Regression test for the catastrophic-domain-shift bug: when some
    sessions have falls (fall-bearing) and others do not (no-fall), the
    routing must not put fall-bearing sessions exclusively in val/test
    and no-fall sessions exclusively in train. Train must contain at
    least some samples from every session that contributed enough
    subjects to be split."""
    samples = []
    # Three fall-bearing lab sessions × 6 subjects each
    for sk in ["dec2017_uog", "feb2019_uog", "jul2018_uog"]:
        for subj in [f"P{i:02d}" for i in range(1, 7)]:
            for act in [1, 2, 3, 4, 5, 6]:
                samples.append(_mock_sample(sk, subj, act))
    # Two no-fall community sessions × 6 subjects each
    for sk in ["feb2019_nghomes", "mar2019_cumbria"]:
        for subj in [f"P{i:02d}" for i in range(20, 26)]:
            for act in [1, 2, 3, 4, 5]:        # 5 activities, no fall
                samples.append(_mock_sample(sk, subj, act))

    splits = subject_disjoint_split(samples, seed=0)
    train_sessions = {s.session.key for s in splits.train}
    val_sessions   = {s.session.key for s in splits.val}
    test_sessions  = {s.session.key for s in splits.test}

    # Every session that has 3+ subjects must contribute to train
    expected = {"dec2017_uog", "feb2019_uog", "jul2018_uog",
                "feb2019_nghomes", "mar2019_cumbria"}
    assert train_sessions == expected, \
        (f"Train sessions {train_sessions} != expected {expected}. "
         f"Domain-shift bug regression: train must include every session.")
    # Falls must end up in val and test (not all in one)
    assert any(s.is_fall for s in splits.val), "No falls in val"
    assert any(s.is_fall for s in splits.test), "No falls in test"
    # And of course zero falls in train
    assert sum(1 for s in splits.train if s.is_fall) == 0


# ----------------------------------------------------------------------
# Activity codes
# ----------------------------------------------------------------------
def test_activity_codes_complete():
    expected = {1: "walking", 2: "sitting_down", 3: "standing_up",
                4: "picking_up", 5: "drinking", 6: "falling"}
    assert ACTIVITY_NAMES == expected
