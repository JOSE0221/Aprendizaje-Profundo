"""
labels.py
=========
Filename parsing for the seven INSHEP sessions.

Canonical pattern (Readme): KPXXAYYRZ.dat where
    K   = activity digit, 1..6
    XX  = subject local id, e.g. 36 (one or two digits, sometimes
          three with leading zero, sometimes prefixed 'P')
    YY  = activity code A01..A06 (often redundant with K)
    Z   = repetition R1, R2, ...

Real-world variations encountered across the seven sessions include:
    1P36A01R1.dat
    1_P36_A01_R1.dat
    1P36A01R01.dat
    1P36 A01 R1.dat       (some sessions use spaces!)
    1p36a01r1.dat         (lowercase)

We accept all of these. The leading digit K is the canonical activity:
the redundant A## code is sometimes inconsistent in the corpus and we
prefer the K digit when they conflict (this is what the Readme
documents as authoritative).
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sessions import Session, session_by_folder, global_subject_id


ACTIVITY_NAMES = {
    1: "walking",
    2: "sitting_down",
    3: "standing_up",
    4: "picking_up",
    5: "drinking",
    6: "falling",
}

NORMAL_ACTIVITIES = frozenset({1, 2, 3, 4, 5})
ANOMALY_ACTIVITY  = 6


# Single forgiving pattern: separators (_, space, or none) between fields
_PATTERN = re.compile(
    r"""
    ^
    (?P<k>[1-6])                       # leading activity digit
    [\s_]*P?(?P<subj>\d{1,3})          # optional 'P' then 1-3 digit subj id
    [\s_]*A(?P<act>\d{1,2})            # 'A' then 1-2 digit activity code
    [\s_]*R(?P<rep>\d{1,2})            # 'R' then 1-2 digit repetition
    \.dat$
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class RadarSample:
    path: Path                 # absolute path to the .dat file
    session: Session           # which collection session
    local_id: str              # canonical "P##" within the session
    activity: int              # 1..6
    repetition: int

    @property
    def subject_id(self) -> str:
        """GLOBAL identifier — unique across the whole corpus."""
        return global_subject_id(self.session.key, self.local_id)

    @property
    def label_name(self) -> str:
        return ACTIVITY_NAMES[self.activity]

    @property
    def is_fall(self) -> bool:
        return self.activity == ANOMALY_ACTIVITY


def _canonical_local_id(raw: str) -> str:
    """Convert '36' or 'P36' or 'p036' into 'P36' canonically."""
    digits = re.sub(r"\D", "", raw)
    return "P" + digits.lstrip("0").zfill(2) if digits else "P00"


def parse_dat_filename(path: str | Path,
                       session: Session) -> Optional[RadarSample]:
    """Parse one .dat filename within a known session. Returns None for
    files that do not match the canonical pattern (the caller should
    log and skip them rather than crash).
    """
    p = Path(path)
    m = _PATTERN.match(p.name)
    if m is None:
        return None
    k = int(m.group("k"))
    if k not in ACTIVITY_NAMES:
        return None
    return RadarSample(
        path=p.absolute(),
        session=session,
        local_id=_canonical_local_id(m.group("subj")),
        activity=k,
        repetition=int(m.group("rep")),
    )


def walk_session(session: Session, session_path: Path) -> list[RadarSample]:
    """Recursively walk a session folder (NG Homes has Room 1/2/3
    subfolders) and return every parseable .dat file."""
    out: list[RadarSample] = []
    for f in sorted(session_path.rglob("*.[dD][aA][tT]")):
        s = parse_dat_filename(f, session)
        if s is not None:
            out.append(s)
    return out


def load_corpus(root: str | Path) -> list[RadarSample]:
    """Walk the entire 'Falling' directory and load all sessions."""
    from sessions import list_session_paths
    out: list[RadarSample] = []
    for sess, p in list_session_paths(root):
        out.extend(walk_session(sess, p))
    return out


def summarize(samples: list[RadarSample]) -> dict:
    by_act = {a: 0 for a in ACTIVITY_NAMES}
    by_session: dict[str, int] = {}
    by_global_subj: dict[str, int] = {}
    for s in samples:
        by_act[s.activity] += 1
        by_session[s.session.key] = by_session.get(s.session.key, 0) + 1
        by_global_subj[s.subject_id] = by_global_subj.get(s.subject_id, 0) + 1
    return {
        "n_files":          len(samples),
        "n_global_subjects": len(by_global_subj),
        "by_activity":      {ACTIVITY_NAMES[k]: v for k, v in by_act.items()},
        "by_session":       by_session,
    }
