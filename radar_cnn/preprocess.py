"""
preprocess.py
-------------
Batch-convert every `.dat` file under ``radar_data/`` into micro-Doppler
spectrograms (one or two per file, depending on the activity) and cache
them on disk as a single compressed ``.npz`` dataset.

Run::

    python preprocess.py                  # uses defaults
    python preprocess.py --workers 8      # parallel preprocessing
    python preprocess.py --target-shape 128 256
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np

from radar_utils import (
    ACTIVITY_NAMES,
    NUM_CLASSES,
    file_to_spectrograms,
    iter_radar_files,
    normalize_spectrogram,
    parse_filename,
)


def _resize_spectrogram(spec: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize a 2-D spectrogram to ``target_shape`` using bilinear interpolation.

    Uses scipy's ``zoom``; fallback to numpy decimation if scipy is unavailable.
    """
    if target_shape is None or tuple(spec.shape) == tuple(target_shape):
        return spec.astype(np.float32)
    try:
        from scipy.ndimage import zoom
        zy = target_shape[0] / spec.shape[0]
        zx = target_shape[1] / spec.shape[1]
        return zoom(spec, (zy, zx), order=1).astype(np.float32)
    except Exception:
        # Nearest-neighbour fallback.
        yi = np.linspace(0, spec.shape[0] - 1, target_shape[0]).astype(int)
        xi = np.linspace(0, spec.shape[1] - 1, target_shape[1]).astype(int)
        return spec[np.ix_(yi, xi)].astype(np.float32)


def _process_one(args) -> Tuple[List[np.ndarray], List[int], List[int], List[str]]:
    """Worker function executed per file (pickle-friendly signature)."""
    path, target_shape, do_normalize = args
    activity, person, _, _ = parse_filename(path)
    results = file_to_spectrograms(path)
    specs: List[np.ndarray] = []
    labels: List[int] = []
    persons: List[int] = []
    names: List[str] = []
    for spec, label in results:
        if target_shape is not None:
            spec = _resize_spectrogram(spec, target_shape)
        if do_normalize:
            spec = normalize_spectrogram(spec)
        specs.append(spec.astype(np.float32))
        labels.append(int(label))
        persons.append(int(person))
        names.append(Path(path).name)
    return specs, labels, persons, names


def build_dataset(data_dir: str | Path,
                  output_path: str | Path,
                  target_shape: Tuple[int, int] | None = (128, 256),
                  normalize: bool = True,
                  workers: int = 1,
                  limit: int | None = None) -> None:
    """Process every radar file and save the dataset as a compressed .npz."""
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = list(iter_radar_files(data_dir))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise RuntimeError(f"No radar files found in {data_dir}")

    print(f"Found {len(files)} files in {data_dir}")
    print(f"Target spectrogram shape: {target_shape}")
    print(f"Workers: {workers}\n")

    all_specs: List[np.ndarray] = []
    all_labels: List[int] = []
    all_persons: List[int] = []
    all_names: List[str] = []

    t0 = time.time()
    tasks = [(str(p), target_shape, normalize) for p in files]

    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            try:
                specs, labels, persons, names = _process_one(task)
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {task[0]}: {exc}", file=sys.stderr)
                continue
            all_specs.extend(specs)
            all_labels.extend(labels)
            all_persons.extend(persons)
            all_names.extend(names)
            if i % 25 == 0 or i == len(tasks):
                print(f"  processed {i}/{len(tasks)} files  "
                      f"({time.time() - t0:.1f}s, {len(all_specs)} samples)")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_process_one, t): t for t in tasks}
            done = 0
            for fut in as_completed(futures):
                task = futures[fut]
                done += 1
                try:
                    specs, labels, persons, names = fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"  [skip] {task[0]}: {exc}", file=sys.stderr)
                    continue
                all_specs.extend(specs)
                all_labels.extend(labels)
                all_persons.extend(persons)
                all_names.extend(names)
                if done % 25 == 0 or done == len(tasks):
                    print(f"  processed {done}/{len(tasks)} files  "
                          f"({time.time() - t0:.1f}s, {len(all_specs)} samples)")

    if not all_specs:
        raise RuntimeError("No spectrograms were produced; aborting.")

    X = np.stack(all_specs, axis=0).astype(np.float32)
    y = np.asarray(all_labels, dtype=np.int64)
    persons = np.asarray(all_persons, dtype=np.int32)
    names = np.asarray(all_names)

    print(f"\nDataset shape: X={X.shape}  y={y.shape}")
    print("Per-class counts:")
    for c in range(NUM_CLASSES):
        print(f"  {c + 1}. {ACTIVITY_NAMES[c + 1]:<28} "
              f"-> {int((y == c).sum())} samples")

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        persons=persons,
        filenames=names,
        class_names=np.array([ACTIVITY_NAMES[i + 1]
                              for i in range(NUM_CLASSES)]),
    )
    mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved {output_path}  ({mb:.1f} MB)")
    print(f"Total time: {time.time() - t0:.1f}s")


def _parse_shape(value: str) -> Tuple[int, int]:
    parts = value.lower().replace("x", ",").replace(" ", ",").split(",")
    parts = [p for p in parts if p]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("shape must be HxW (e.g. 128x256)")
    return int(parts[0]), int(parts[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="radar_data",
                    help="Directory with the .dat files")
    ap.add_argument("--output", default="processed/dataset.npz",
                    help="Path to the output .npz cache")
    ap.add_argument("--shape", type=_parse_shape, default="128x256",
                    help="Target spectrogram shape as HxW (e.g. 128x256)")
    ap.add_argument("--no-normalize", action="store_true",
                    help="Do not rescale spectrograms to [0, 1]")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1),
                    help="Number of parallel processes")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N files (for debugging)")
    args = ap.parse_args()

    build_dataset(
        data_dir=args.data_dir,
        output_path=args.output,
        target_shape=args.shape,
        normalize=not args.no_normalize,
        workers=args.workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
