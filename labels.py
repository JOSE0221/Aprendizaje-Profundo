"""Parse subject ID and activity class from Glasgow / INSHEP filenames."""

from __future__ import annotations

import re
from pathlib import Path


# INSHEP / data sheet: 1 walk, 2 sit, 3 stand, 4 pick, 5 drink, 6 fall
ACTIVITY_NAMES = {
    1: "walk",
    2: "sit",
    3: "stand",
    4: "pick",
    5: "drink",
    6: "fall",
}

# 0-based PyTorch label for activity A06 (fall); see activity_to_label(6).
FALL_CLASS_INDEX = 5


def parse_filename(path: str | Path) -> tuple[int, int, int]:
    """
    Parse subject id, activity class (1-6), repetition index from filename.

    Supports patterns like: 1P01A01R2.dat, 2P38A02R01.dat (optional leading digit).
    Activity is taken from A01..A06 (primary) or leading K digit if present.
    """
    stem = Path(path).stem
    # Optional leading digit(s) before P
    m = re.search(r"P(\d+)A(\d+)R(\d+)", stem, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse P/A/R from: {path}")
    subject_id = int(m.group(1))
    activity = int(m.group(2))
    rep = int(m.group(3))
    if activity < 1 or activity > 6:
        raise ValueError(f"Activity out of range 1-6: {activity} in {path}")
    return subject_id, activity, rep


def activity_to_label(activity: int) -> int:
    """0-based class index for PyTorch (0..5)."""
    if activity < 1 or activity > 6:
        raise ValueError(activity)
    return activity - 1


def activity_to_binary_label(activity: int) -> int:
    """Binary label: fall=1 (A06), non-fall=0 (A01..A05)."""
    if activity < 1 or activity > 6:
        raise ValueError(activity)
    return 1 if activity == 6 else 0
