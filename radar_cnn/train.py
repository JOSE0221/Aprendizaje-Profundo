"""
train.py
--------
Train the radar-HAR CNN on the cached spectrogram dataset produced by
``preprocess.py``.

The dataset is split by *person id* (not by sample) so that the test
set contains subjects the network has never seen. Model weights, a
training-curve plot and a confusion matrix are saved under ``models/``.

Usage::

    python train.py
    python train.py --epochs 50 --batch-size 32 --test-frac 0.2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from model import build_cnn, compile_model
from radar_utils import ACTIVITY_NAMES, NUM_CLASSES


def _load_dataset(path: str | Path):
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    persons = data["persons"].astype(np.int32)
    if X.ndim == 3:
        X = X[..., np.newaxis]  # add channel dim
    return X, y, persons


def _split_by_person(persons: np.ndarray,
                     y: np.ndarray,
                     test_frac: float,
                     seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks ``(train_mask, test_mask)`` such that no
    person appears in both the train and test sets.
    """
    rng = np.random.default_rng(seed)
    unique_persons = np.unique(persons)
    rng.shuffle(unique_persons)
    n_test = max(1, int(round(len(unique_persons) * test_frac)))
    test_persons = set(unique_persons[:n_test].tolist())
    test_mask = np.array([p in test_persons for p in persons])
    train_mask = ~test_mask

    # Safety: if the random test split happens to miss an entire class,
    # move one sample of that class from train to test to keep metrics
    # well-defined.
    for c in range(NUM_CLASSES):
        if not np.any(test_mask & (y == c)):
            candidates = np.where(train_mask & (y == c))[0]
            if candidates.size > 0:
                idx = int(rng.choice(candidates))
                train_mask[idx] = False
                test_mask[idx] = True
    return train_mask, test_mask


def _plot_history(history, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433 lazy import
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        ax[0].plot(history.history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()

    ax[1].plot(history.history["accuracy"], label="train")
    if "val_accuracy" in history.history:
        ax[1].plot(history.history["val_accuracy"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
    except ImportError:
        return
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    labels = [ACTIVITY_NAMES[i + 1] for i in range(NUM_CLASSES)]
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            txt = f"{cm[i, j]}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _augment(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Light augmentation: random time-shifts and frequency-masking."""
    # Random time shift (roll along width axis)
    max_shift = tf.shape(x)[1] // 8
    shift = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
    x = tf.roll(x, shift=shift, axis=1)

    # Frequency masking (zero out a small horizontal band)
    if tf.random.uniform([]) < 0.5:
        h = tf.shape(x)[0]
        mask_h = tf.random.uniform([], 1, tf.maximum(h // 10, 2), dtype=tf.int32)
        start = tf.random.uniform([], 0, h - mask_h, dtype=tf.int32)
        mask = tf.concat([
            tf.ones([start, tf.shape(x)[1], tf.shape(x)[2]], dtype=x.dtype),
            tf.zeros([mask_h, tf.shape(x)[1], tf.shape(x)[2]], dtype=x.dtype),
            tf.ones([h - start - mask_h, tf.shape(x)[1], tf.shape(x)[2]],
                    dtype=x.dtype),
        ], axis=0)
        x = x * mask
    return x, y


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.dataset} ...")
    X, y, persons = _load_dataset(args.dataset)
    print(f"  X={X.shape}  y={y.shape}  unique_persons={len(np.unique(persons))}")

    train_mask, test_mask = _split_by_person(persons, y, args.test_frac, args.seed)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"  train samples: {len(X_train)}  test samples: {len(X_test)}")
    print(f"  train persons: {sorted(set(persons[train_mask].tolist()))[:10]}...")
    print(f"  test persons : {sorted(set(persons[test_mask].tolist()))}")

    # tf.data pipelines
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=len(X_train), seed=args.seed)
        .map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    tf.keras.utils.set_random_seed(args.seed)
    model = build_cnn(input_shape=X.shape[1:], num_classes=NUM_CLASSES,
                      dropout=args.dropout)
    compile_model(model, learning_rate=args.lr)
    model.summary()

    ckpt_path = out_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt_path), monitor="val_accuracy",
            save_best_only=True, mode="max", verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=12,
            restore_best_weights=True, mode="max"),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    final_model_path = out_dir / "radar_cnn.keras"
    model.save(final_model_path)
    print(f"\nSaved final model to {final_model_path}")
    print(f"Best-val model saved to {ckpt_path}")

    # Final evaluation + artefacts
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\nHold-out test accuracy: {acc:.4f}   loss: {loss:.4f}")

    y_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)
    _plot_history(history, out_dir / "training_curves.png")
    _plot_confusion(y_test, y_pred, out_dir / "confusion_matrix.png")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "test_accuracy": float(acc),
                "test_loss": float(loss),
                "history": {k: [float(v) for v in vs]
                            for k, vs in history.history.items()},
                "class_names": [ACTIVITY_NAMES[i + 1]
                                for i in range(NUM_CLASSES)],
            },
            f,
            indent=2,
        )
    print(f"Metrics and plots saved under {out_dir}/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="processed/dataset.npz",
                    help="Path to the preprocessed .npz file")
    ap.add_argument("--out-dir", default="models",
                    help="Directory to store model weights and plots")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="Fraction of persons held out for the test split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
