"""
demographics.py
===============
Per-(session, local_id) demographic metadata transcribed verbatim from
the Readme tables. We hardcode rather than parse the PDF because the
PDF tables contain typos, missing cells ('n/a'), and inconsistent
formatting; a hand-curated table is more reliable.

Schema:
    age          : int | None     # years
    height_cm    : float | None   # cm; many West Cumbria entries are n/a
    dominant_hand: 'R' | 'L' | None
    gender       : 'M' | 'F'

Coverage warning: not every subject has every field. Code that consumes
this should handle Nones gracefully.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Demographics:
    age: Optional[int]
    height_cm: Optional[float]
    dominant_hand: Optional[str]   # 'R' / 'L' / None
    gender: str                     # 'M' / 'F'


def _R(age, h, hand, g):
    return Demographics(age=age, height_cm=h, dominant_hand=hand, gender=g)


# Keys are (session_key, local_id). local_id is the canonical "P##" string.
DEMOGRAPHICS: dict[tuple[str, str], Demographics] = {
    # =========================================================
    # 1 — December 2017 (UoG laboratory) — 20 subjects P36..P56 minus P49
    # =========================================================
    ("dec2017_uog", "P36"): _R(27, 182,  "R", "M"),
    ("dec2017_uog", "P37"): _R(27, 176,  "R", "M"),
    ("dec2017_uog", "P38"): _R(28, 182,  "R", "M"),
    ("dec2017_uog", "P39"): _R(23, 182,  "R", "M"),
    ("dec2017_uog", "P40"): _R(22, 183,  "R", "M"),
    ("dec2017_uog", "P41"): _R(23, 185, None, "M"),
    ("dec2017_uog", "P42"): _R(22, 180,  "R", "M"),
    ("dec2017_uog", "P43"): _R(25, 181,  "R", "M"),
    ("dec2017_uog", "P44"): _R(27, 167,  "R", "M"),
    ("dec2017_uog", "P45"): _R(25, 173,  "R", "M"),
    ("dec2017_uog", "P46"): _R(31, 167,  "R", "M"),
    ("dec2017_uog", "P47"): _R(27, 180,  "R", "M"),
    ("dec2017_uog", "P48"): _R(34, 172,  "R", "M"),
    ("dec2017_uog", "P50"): _R(24, 182,  "R", "M"),
    ("dec2017_uog", "P51"): _R(26, 178,  "R", "M"),
    ("dec2017_uog", "P52"): _R(26, 170,  "R", "M"),
    ("dec2017_uog", "P53"): _R(21, 180, None, "M"),
    ("dec2017_uog", "P54"): _R(21, 180,  "R", "M"),
    ("dec2017_uog", "P55"): _R(23, 188,  "R", "M"),
    ("dec2017_uog", "P56"): _R(32, 170,  "R", "M"),

    # =========================================================
    # 2 — March 2017 (UoG laboratory)
    # =========================================================
    ("mar2017_uog", "P03"): _R(23, 180,  "R", "M"),
    ("mar2017_uog", "P10"): _R(23, 182,  "R", "M"),
    ("mar2017_uog", "P11"): _R(23, 182,  "R", "M"),
    ("mar2017_uog", "P12"): _R(31, 170,  "R", "M"),

    # =========================================================
    # 3 — June 2017 (UoG laboratory)
    # =========================================================
    ("jun2017_uog", "P14"): _R(None, None, None, "F"),
    ("jun2017_uog", "P28"): _R(27, 180,  "L", "M"),
    ("jun2017_uog", "P29"): _R(27, 176,  "R", "M"),
    ("jun2017_uog", "P30"): _R(23, 180,  "R", "M"),
    ("jun2017_uog", "P31"): _R(23, 149,  "R", "F"),
    ("jun2017_uog", "P32"): _R(26, 173,  "R", "M"),
    ("jun2017_uog", "P33"): _R(24, 173,  "R", "M"),
    # P34 has no age in Readme; height 176, left hand
    ("jun2017_uog", "P34"): _R(None, 176, "L", "M"),
    ("jun2017_uog", "P35"): _R(36, 175,  "R", "M"),

    # =========================================================
    # 4 — July 2018 (UoG common room)
    # =========================================================
    ("jul2018_uog", "P57"): _R(32, 170,  "R", "M"),
    ("jul2018_uog", "P58"): _R(25, 168,  "R", "F"),
    ("jul2018_uog", "P59"): _R(32, 168,  "L", "M"),
    ("jul2018_uog", "P60"): _R(25, 170,  "R", "M"),
    ("jul2018_uog", "P61"): _R(27, 173,  "R", "M"),
    ("jul2018_uog", "P62"): _R(26, 173,  "R", "M"),
    ("jul2018_uog", "P63"): _R(27, 178,  "R", "M"),
    ("jul2018_uog", "P64"): _R(28, 177,  "R", "M"),
    ("jul2018_uog", "P65"): _R(23, 180,  "R", "M"),
    ("jul2018_uog", "P66"): _R(26, 180,  "R", "M"),
    ("jul2018_uog", "P67"): _R(27, 165,  "R", "M"),
    ("jul2018_uog", "P68"): _R(25, 180,  "R", "M"),
    ("jul2018_uog", "P69"): _R(36, 182,  "R", "M"),
    ("jul2018_uog", "P70"): _R(26, 180,  "R", "M"),
    ("jul2018_uog", "P71"): _R(24, 178,  "R", "M"),
    ("jul2018_uog", "P72"): _R(28, 168,  "R", "M"),

    # =========================================================
    # 5 — February 2019 UoG laboratory
    # =========================================================
    ("feb2019_uog", "P01"): _R(25, 180,  "R", "M"),
    ("feb2019_uog", "P02"): _R(37, 182,  "R", "M"),
    ("feb2019_uog", "P03"): _R(32, 183,  "R", "M"),  # different from mar2017 P03
    ("feb2019_uog", "P04"): _R(36, 170,  "R", "M"),
    ("feb2019_uog", "P05"): _R(31, 170,  "R", "M"),
    ("feb2019_uog", "P06"): _R(44, 177,  "R", "M"),
    ("feb2019_uog", "P07"): _R(34, 165,  "R", "F"),
    ("feb2019_uog", "P08"): _R(33, 170,  "R", "M"),  # also appears in NG Homes
    ("feb2019_uog", "P09"): _R(30, 167,  "R", "M"),
    ("feb2019_uog", "P10"): _R(27, 173,  "R", "M"),  # different from mar2017 P10
    ("feb2019_uog", "P11"): _R(25, 161,  "R", "F"),  # different from mar2017 P11
    ("feb2019_uog", "P12"): _R(25, 182,  "L", "M"),  # different from mar2017 P12
    ("feb2019_uog", "P13"): _R(31, 179,  "R", "M"),
    ("feb2019_uog", "P14"): _R(32, 168,  "R", "M"),  # different from jun2017 P14
    ("feb2019_uog", "P15"): _R(27, 181,  "R", "M"),
    ("feb2019_uog", "P16"): _R(25, 180,  "R", "M"),
    ("feb2019_uog", "P17"): _R(25, 180,  "R", "M"),

    # =========================================================
    # 6 — February 2019 NG Homes (no falls in this session)
    # =========================================================
    # Room 1
    ("feb2019_nghomes", "P08"): _R(33, 170,   "R", "M"),  # plausibly same as feb2019_uog P08
    ("feb2019_nghomes", "P18"): _R(65, 164.5, "L", "M"),
    ("feb2019_nghomes", "P19"): _R(82, 170.6, "R", "M"),
    ("feb2019_nghomes", "P20"): _R(78, 170.6, "R", "M"),
    ("feb2019_nghomes", "P21"): _R(66, 152.4, "R", "M"),  # missing 1 walk rep
    ("feb2019_nghomes", "P22"): _R(33, 161.5, "R", "F"),
    ("feb2019_nghomes", "P23"): _R(50, 155.5, "R", "M"),  # missing 1 walk rep
    ("feb2019_nghomes", "P24"): _R(56, 152.4, "R", "M"),
    ("feb2019_nghomes", "P25"): _R(25, 155.4, "R", "F"),
    # Room 2
    ("feb2019_nghomes", "P26"): _R(88, None,  "R", "M"),
    ("feb2019_nghomes", "P27"): _R(63, 176.7, "R", "M"),
    ("feb2019_nghomes", "P28"): _R(79, 176.7, "L", "M"),  # different from jun2017 P28
    ("feb2019_nghomes", "P29"): _R(68, None,  "R", "F"),  # different from jun2017 P29
    ("feb2019_nghomes", "P30"): _R(65, 178.3, "R", "M"),  # different from jun2017 P30
    ("feb2019_nghomes", "P31"): _R(24, None,  "R", "M"),  # different from jun2017 P31
    ("feb2019_nghomes", "P32"): _R(84, None,  "R", "M"),  # different from jun2017 P32
    # Room 3
    ("feb2019_nghomes", "P33"): _R(79, None,  "R", "F"),  # different from jun2017 P33
    ("feb2019_nghomes", "P34"): _R(60, None,  "L", "F"),  # different from jun2017 P34
    ("feb2019_nghomes", "P35"): _R(64, None,  "R", "F"),  # different from jun2017 P35
    ("feb2019_nghomes", "P36"): _R(70, None,  "R", "F"),  # different from dec2017 P36

    # =========================================================
    # 7 — March 2019 West Cumbria (Age UK community centre)
    # =========================================================
    ("mar2019_cumbria", "P37"): _R(75, None, "R", "F"),
    ("mar2019_cumbria", "P38"): _R(74, None, "R", "F"),
    ("mar2019_cumbria", "P39"): _R(52, None, "L", "M"),
    ("mar2019_cumbria", "P40"): _R(48, None, "R", "F"),
    ("mar2019_cumbria", "P41"): _R(84, None, "R", "F"),
    ("mar2019_cumbria", "P42"): _R(85, None, "R", "M"),  # ⚠ limited data
    ("mar2019_cumbria", "P43"): _R(67, None, "R", "M"),
    ("mar2019_cumbria", "P44"): _R(45, None, "R", "F"),
    ("mar2019_cumbria", "P45"): _R(78, None, "R", "F"),
    ("mar2019_cumbria", "P46"): _R(67, None, "R", "F"),
    ("mar2019_cumbria", "P47"): _R(98, None, "R", "F"),
    ("mar2019_cumbria", "P48"): _R(47, None, "R", "F"),
    ("mar2019_cumbria", "P49"): _R(57, None, "R", "M"),
    ("mar2019_cumbria", "P50"): _R(71, None, "R", "F"),
    ("mar2019_cumbria", "P51"): _R(50, None, "R", "F"),
    ("mar2019_cumbria", "P52"): _R(49, None, "R", "F"),
    ("mar2019_cumbria", "P53"): _R(84, None, "R", "M"),
    ("mar2019_cumbria", "P54"): _R(69, None, "R", "F"),
    ("mar2019_cumbria", "P55"): _R(57, None, "R", "F"),
    ("mar2019_cumbria", "P56"): _R(25, None, "R", "F"),  # different from dec2017 P56
}


# Subjects flagged in the Readme as having data quality issues
DATA_QUALITY_FLAGS = {
    ("feb2019_nghomes", "P21"): "missing one A01 walk repetition",
    ("feb2019_nghomes", "P23"): "missing one A01 walk repetition",
    ("feb2019_nghomes", "P08"): "three extra A06 fall repetitions recorded "
                                "in NG Homes (UNUSUAL: this session normally "
                                "has no falls)",
    ("mar2019_cumbria",  "P42"): "limited data — may not have all reps",
}


def get_demographics(session_key: str, local_id: str) -> Optional[Demographics]:
    return DEMOGRAPHICS.get((session_key, local_id))


def is_elderly(session_key: str, local_id: str, threshold_age: int = 60) -> bool:
    """Convenience flag for fairness-audit reporting."""
    d = get_demographics(session_key, local_id)
    return bool(d and d.age is not None and d.age >= threshold_age)
