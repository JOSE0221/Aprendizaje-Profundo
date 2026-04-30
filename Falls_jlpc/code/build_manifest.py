#!/usr/bin/env python3
"""
build_manifest.py
-------------------------
Walk all seven INSHEP session folders, parse every .dat file, join with
the demographic table, and write a master manifest.csv.

Default --data_root is "..", which makes the script work out-of-the-box
when run from inside Falling/code/.

Run:
    python build_manifest.py
    python build_manifest.py --data_root /path/to/Falling --out manifest.csv
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path


from manifest import build_manifest, write_manifest, manifest_summary
from sessions import list_session_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="..",
                    help="Path to the 'Falling' directory containing the "
                         "seven session folders (default: '..' since this "
                         "script lives in Falling/code/).")
    ap.add_argument("--out", default="manifest.csv")
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    print(f"[manifest] data_root = {root}")

    # Validate folder layout up front — fail loud, not silent.
    found = list_session_paths(root)
    print(f"[manifest] found all 7 session folders:")
    for s, p in found:
        print(f"           {s.key:18s} → {p}")

    rows = build_manifest(root)
    write_manifest(rows, args.out)
    print(f"[manifest] wrote {len(rows)} rows → {args.out}")

    summary = manifest_summary(rows)
    print("\n[manifest] per-session counts vs. expected:")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
