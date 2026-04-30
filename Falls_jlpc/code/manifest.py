"""
manifest.py
===========
Build a single CSV manifest joining every .dat file in the corpus with
its parsed labels, the (session, local_id) demographics, and any data-
quality flags from the Readme. Downstream code consumes the manifest;
no other module should re-walk the filesystem.

Usage (script form):
    python -m scripts.build_manifest --data_root .. --out manifest.csv
"""

from __future__ import annotations
import csv
from pathlib import Path

from labels import load_corpus, ACTIVITY_NAMES
from demographics import get_demographics, DATA_QUALITY_FLAGS


CSV_FIELDS = [
    "path",
    "session_key",
    "session_folder",
    "location",
    "local_id",
    "global_subject_id",
    "activity_code",
    "activity_name",
    "repetition",
    "is_fall",
    "age",
    "height_cm",
    "dominant_hand",
    "gender",
    "is_elderly",
    "quality_flag",
]


def build_manifest(root: str | Path) -> list[dict]:
    samples = load_corpus(root)
    rows: list[dict] = []
    for s in samples:
        d = get_demographics(s.session.key, s.local_id)
        rows.append({
            "path":              str(s.path),
            "session_key":       s.session.key,
            "session_folder":    s.session.folder,
            "location":          s.session.location,
            "local_id":          s.local_id,
            "global_subject_id": s.subject_id,
            "activity_code":     s.activity,
            "activity_name":     s.label_name,
            "repetition":        s.repetition,
            "is_fall":           int(s.is_fall),
            "age":               d.age          if d else None,
            "height_cm":         d.height_cm    if d else None,
            "dominant_hand":     d.dominant_hand if d else None,
            "gender":            d.gender       if d else None,
            "is_elderly":        int(bool(d and d.age and d.age >= 60)),
            "quality_flag":      DATA_QUALITY_FLAGS.get(
                                   (s.session.key, s.local_id), ""),
        })
    return rows


def write_manifest(rows: list[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_manifest(path: str | Path) -> list[dict]:
    """Read back a manifest CSV. Numeric columns are restored to int/float."""
    rows: list[dict] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["activity_code"] = int(r["activity_code"])
            r["repetition"]    = int(r["repetition"])
            r["is_fall"]       = bool(int(r["is_fall"]))
            r["is_elderly"]    = bool(int(r["is_elderly"]))
            r["age"]           = int(r["age"]) if r["age"] not in ("", "None") else None
            r["height_cm"]     = float(r["height_cm"]) if r["height_cm"] not in ("", "None") else None
            r["dominant_hand"] = r["dominant_hand"] if r["dominant_hand"] not in ("", "None") else None
            rows.append(r)
    return rows


def manifest_summary(rows: list[dict]) -> dict:
    """Quick QA dashboard. Run after build_manifest() and check that
    counts match the Readme's expected_files per session.

    Each per-session block carries a `delta` and a `notes` field that
    explains common deviations (e.g., '1 file missing — check disk').
    """
    from sessions import SESSIONS

    by_sess: dict[str, dict[str, int]] = {}
    for r in rows:
        sk = r["session_key"]
        d = by_sess.setdefault(sk, {"total": 0, "falls": 0, "subjects": set()})
        d["total"] += 1
        d["falls"] += int(r["is_fall"])
        d["subjects"].add(r["local_id"])

    summary = {}
    for sk, counts in by_sess.items():
        sess = SESSIONS[sk]
        delta_files = counts["total"] - sess.expected_files
        notes = []
        if delta_files != 0:
            notes.append(f"{abs(delta_files)} file(s) "
                         f"{'missing' if delta_files < 0 else 'extra'}")
        if not sess.has_falls and counts["falls"] > 0:
            notes.append(f"{counts['falls']} unexpected fall file(s) "
                         f"in session marked has_falls=False")
        summary[sk] = {
            "actual_files":      counts["total"],
            "expected_files":    sess.expected_files,
            "delta_files":       delta_files,
            "actual_subjects":   len(counts["subjects"]),
            "expected_subjects": sess.expected_subjects,
            "actual_falls":      counts["falls"],
            "expected_falls":    None if not sess.has_falls
                                  else "varies (~3 reps × n subjects)",
            "match": (counts["total"] == sess.expected_files
                      and len(counts["subjects"]) == sess.expected_subjects),
            "notes": "; ".join(notes) if notes else "ok",
        }
    summary["TOTAL"] = {
        "files":            len(rows),
        "global_subjects":  len({r["global_subject_id"] for r in rows}),
        "falls":            sum(r["is_fall"] for r in rows),
        "elderly_files":    sum(r["is_elderly"] for r in rows),
        "female_files":     sum(1 for r in rows if r["gender"] == "F"),
    }
    return summary
