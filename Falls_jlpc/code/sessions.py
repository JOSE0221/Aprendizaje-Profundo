"""
sessions.py
===========
Canonical metadata for the seven INSHEP collection sessions, hardcoded to
match the EXACT folder names visible in the dataset directory:

    Falling/
      ├── 1 December 2017 Dataset/
      ├── 2 March 2017 Dataset/
      ├── 3 June 2017 Dataset/
      ├── 4 July 2018 Dataset/
      ├── 5 February 2019 UoG Dataset/
      ├── 6 February 2019 NG Homes Dataset/
      ├── 7 March 2019 West Cumbria Dataset/
      ├── code/                          (this package lives here)
      ├── AutoEncoders.pdf
      └── Document for all datasets - 03092019.pdf

Why this module exists
----------------------
1. Subject IDs are NOT unique across sessions. P36 in `1 December 2017` is
   a 27-year-old male; P36 in `6 February 2019 NG Homes` is a 70-year-old
   female. They are different people. The canonical subject identifier
   must therefore be (session_key, local_id), not local_id alone.

2. NG Homes (session 6) recorded only 5 activities — no falls. Any code
   that tries to find fall samples in this folder will silently produce
   empty results unless told ahead of time.

3. The folder names start with digits and contain spaces. pathlib handles
   this fine but shell scripts and globs need careful quoting.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Session:
    key: str                      # short stable id used everywhere downstream
    folder: str                   # exact folder name on disk
    location: str                 # human-readable site
    expected_files: int           # from the Readme
    expected_subjects: int
    expected_repetitions: int
    has_falls: bool               # NG Homes does NOT have A06
    notes: str = ""

    def path(self, root: Path | str) -> Path:
        return Path(root) / self.folder


SESSIONS: dict[str, Session] = {
    "dec2017_uog": Session(
        key="dec2017_uog",
        folder="1 December 2017 Dataset",
        location="University of Glasgow laboratory room",
        expected_files=360, expected_subjects=20, expected_repetitions=3,
        has_falls=True,
        notes="Subjects P36–P56 (skipping P49). 6 activities × 20 × 3 = 360."
    ),
    "mar2017_uog": Session(
        key="mar2017_uog",
        folder="2 March 2017 Dataset",
        location="University of Glasgow laboratory room",
        expected_files=48, expected_subjects=4, expected_repetitions=2,
        has_falls=True,
        notes="Subjects P03, P10, P11, P12. Note: these P-numbers also appear "
              "in Feb 2019 UoG with DIFFERENT demographics — different people."
    ),
    "jun2017_uog": Session(
        key="jun2017_uog",
        folder="3 June 2017 Dataset",
        location="University of Glasgow laboratory room",
        expected_files=162, expected_subjects=9, expected_repetitions=3,
        has_falls=True,
        notes="Subjects P14, P28–P35. Multiple ID overlaps with later sessions."
    ),
    "jul2018_uog": Session(
        key="jul2018_uog",
        folder="4 July 2018 Dataset",
        location="University of Glasgow common room",
        expected_files=288, expected_subjects=16, expected_repetitions=3,
        has_falls=True,
        notes="Subjects P57–P72."
    ),
    "feb2019_uog": Session(
        key="feb2019_uog",
        folder="5 February 2019 UoG Dataset",
        location="University of Glasgow laboratory room",
        expected_files=306, expected_subjects=17, expected_repetitions=3,
        has_falls=True,
        notes="Subjects P01–P17."
    ),
    "feb2019_nghomes": Session(
        key="feb2019_nghomes",
        folder="6 February 2019 NG Homes Dataset",
        location="Glasgow NG Homes (community housing, three rooms)",
        expected_files=301, expected_subjects=20, expected_repetitions=3,
        has_falls=False,                  # ⚠ no A06 in this session
        notes="Subjects P08, P18–P36 (overlap of P-IDs with Dec 2017 and Feb "
              "2019 UoG, but DIFFERENT people). 5 activities only — no falls. "
              "Caveats: P21 and P23 missing one walk repetition; P08 has three "
              "extra falling repetitions ONLY IF the Feb 2019 UoG P08 is the "
              "same person (the Readme suggests so but is not explicit)."
    ),
    "mar2019_cumbria": Session(
        key="mar2019_cumbria",
        folder="7 March 2019 West Cumbria Dataset",
        location="Age UK West Cumbria community centre",
        expected_files=289, expected_subjects=20, expected_repetitions=3,
        has_falls=False,                  # ⚠ confirmed empirically: 0 fall files
        notes="Subjects P37–P56. Same P-IDs as Dec 2017 but different people "
              "(WC P37 is 75y female; Dec 2017 P37 is 27y male). 5 activities "
              "only (no falls) — same protocol as NG Homes for elderly safety. "
              "20 × 5 × 3 = 300 expected; 289 actual reflects P42's limited data "
              "and a few additional missing repetitions across other subjects."
    ),
}


# Order to iterate when building the manifest — by collection date,
# matching the numbered prefixes in the folder names.
SESSION_ORDER = [
    "dec2017_uog",
    "mar2017_uog",
    "jun2017_uog",
    "jul2018_uog",
    "feb2019_uog",
    "feb2019_nghomes",
    "mar2019_cumbria",
]


def session_by_folder(folder_name: str) -> Optional[Session]:
    """Reverse-lookup a Session given its on-disk folder name."""
    for s in SESSIONS.values():
        if s.folder == folder_name:
            return s
    return None


def list_session_paths(root: Path | str) -> list[tuple[Session, Path]]:
    """Return (session, absolute_path) tuples in canonical order, raising
    a clear error if any expected folder is missing.
    """
    root = Path(root)
    out: list[tuple[Session, Path]] = []
    missing: list[str] = []
    for k in SESSION_ORDER:
        s = SESSIONS[k]
        p = s.path(root)
        if not p.is_dir():
            missing.append(s.folder)
        else:
            out.append((s, p))
    if missing:
        raise FileNotFoundError(
            "Missing session folder(s) under " + str(root) + ":\n  - " +
            "\n  - ".join(missing) +
            "\n\nExpected layout:\n  Falling/\n    1 December 2017 Dataset/\n"
            "    2 March 2017 Dataset/\n    ...\n    code/   (this package)"
        )
    return out


def global_subject_id(session_key: str, local_id: str) -> str:
    """The unique identifier across the whole corpus. Always use this for
    splits, NEVER `local_id` alone."""
    return f"{session_key}::{local_id}"
