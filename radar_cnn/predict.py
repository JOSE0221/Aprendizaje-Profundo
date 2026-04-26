"""
predict.py
----------
Classify the human activity recorded in a radar `.dat` file using a
previously-trained CNN (saved by ``train.py``).

Usage::

    python predict.py radar_data/5P01A05R1.dat
    python predict.py some_file.dat --model models/best_model.keras
    python predict.py --show-spectrogram some_file.dat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from radar_utils import (
    ACTIVITY_NAMES,
    NUM_CLASSES,
    file_to_spectrograms,
    normalize_spectrogram,
    parse_filename,
)


def _resize(spec: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if tuple(spec.shape) == tuple(target_shape):
        return spec.astype(np.float32)
    try:
        from scipy.ndimage import zoom
        zy = target_shape[0] / spec.shape[0]
        zx = target_shape[1] / spec.shape[1]
        return zoom(spec, (zy, zx), order=1).astype(np.float32)
    except Exception:
        yi = np.linspace(0, spec.shape[0] - 1, target_shape[0]).astype(int)
        xi = np.linspace(0, spec.shape[1] - 1, target_shape[1]).astype(int)
        return spec[np.ix_(yi, xi)].astype(np.float32)


def prepare_inputs(path: str | Path,
                   input_shape: Tuple[int, int, int]
                   ) -> np.ndarray:
    """Convert a `.dat` file into a batch of CNN inputs (1 or 2 segments)."""
    results = file_to_spectrograms(path)
    if not results:
        raise RuntimeError(f"No spectrogram could be extracted from {path}.")
    h, w = input_shape[0], input_shape[1]
    specs = []
    for spec, _ in results:
        spec = _resize(spec, (h, w))
        spec = normalize_spectrogram(spec)
        specs.append(spec)
    X = np.stack(specs, axis=0).astype(np.float32)
    if X.ndim == 3:
        X = X[..., np.newaxis]
    return X


def predict_file(path: str | Path,
                 model: tf.keras.Model,
                 show_truth: bool = True
                 ) -> Tuple[int, float, np.ndarray]:
    """Run the CNN on every segment of ``path`` and aggregate the probabilities.

    Returns ``(predicted_activity_index_1_based, confidence, per_segment_probs)``.
    """
    input_shape = tuple(model.input_shape[1:])
    X = prepare_inputs(path, input_shape)
    probs = model.predict(X, verbose=0)        # (n_segments, num_classes)
    avg = np.mean(probs, axis=0)               # aggregate segments
    pred = int(np.argmax(avg))
    confidence = float(avg[pred])

    print(f"\nFile: {Path(path).name}")
    try:
        true_act, person, _, rep = parse_filename(path)
        if show_truth:
            print(f"  Filename says: activity={true_act} "
                  f"({ACTIVITY_NAMES[true_act]}), person={person}, rep={rep}")
    except ValueError:
        pass

    print(f"  Segments analysed: {X.shape[0]}")
    print(f"  Prediction: activity {pred + 1} "
          f"({ACTIVITY_NAMES[pred + 1]})   confidence={confidence:.3f}")
    print("  Class probabilities:")
    for c in range(NUM_CLASSES):
        bar = "#" * int(avg[c] * 30)
        print(f"    {c + 1}. {ACTIVITY_NAMES[c + 1]:<28} "
              f"{avg[c]:.3f}  {bar}")
    return pred + 1, confidence, probs


def _maybe_plot(path: Path, model: tf.keras.Model) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
    except ImportError:
        print("(matplotlib not installed; skipping spectrogram plot)")
        return
    input_shape = tuple(model.input_shape[1:])
    X = prepare_inputs(path, input_shape)
    n = X.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for i in range(n):
        ax = axes[0, i]
        ax.imshow(X[i, ..., 0], aspect="auto", origin="lower", cmap="jet")
        ax.set_title(f"{path.name}  (segment {i + 1})")
        ax.set_xlabel("time frame"); ax.set_ylabel("Doppler bin")
    fig.tight_layout()
    out = path.with_suffix(".spectrogram.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Spectrogram saved to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", help="Path to a radar .dat file")
    ap.add_argument("--model", default="models/best_model.keras",
                    help="Trained Keras model to use")
    ap.add_argument("--show-spectrogram", action="store_true",
                    help="Save the spectrogram as a PNG next to the input file")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}\n"
              f"Train one first with: python train.py",
              file=sys.stderr)
        sys.exit(1)

    model = tf.keras.models.load_model(model_path, compile=False)
    predict_file(args.file, model)
    if args.show_spectrogram:
        _maybe_plot(Path(args.file), model)


if __name__ == "__main__":
    main()
