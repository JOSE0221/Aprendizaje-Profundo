"""
model.py
--------
CNN architecture used to classify human activities from radar
micro-Doppler spectrograms.

The network is a compact VGG-style stack tailored to single-channel
spectrogram inputs: four convolutional blocks with batch-norm and
max-pooling, followed by a global-average-pooled dense classifier.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_cnn(input_shape: Tuple[int, int, int],
              num_classes: int = 6,
              dropout: float = 0.5,
              weight_decay: float = 1e-4) -> tf.keras.Model:
    """Build and return the radar-HAR CNN.

    Parameters
    ----------
    input_shape : (H, W, C)
        Shape of a single spectrogram. ``C`` is 1 for magnitude-only
        micro-Doppler spectrograms.
    num_classes : int
        Number of activities (6 by default).
    """
    reg = regularizers.l2(weight_decay)

    inputs = layers.Input(shape=input_shape, name="spectrogram")

    def conv_block(x, filters, block):
        x = layers.Conv2D(filters, 3, padding="same",
                          kernel_regularizer=reg,
                          name=f"b{block}_conv1")(x)
        x = layers.BatchNormalization(name=f"b{block}_bn1")(x)
        x = layers.ReLU(name=f"b{block}_relu1")(x)
        x = layers.Conv2D(filters, 3, padding="same",
                          kernel_regularizer=reg,
                          name=f"b{block}_conv2")(x)
        x = layers.BatchNormalization(name=f"b{block}_bn2")(x)
        x = layers.ReLU(name=f"b{block}_relu2")(x)
        x = layers.MaxPooling2D(pool_size=2, name=f"b{block}_pool")(x)
        return x

    x = conv_block(inputs, 32, 1)
    x = conv_block(x, 64, 2)
    x = conv_block(x, 128, 3)
    x = conv_block(x, 128, 4)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=reg, name="fc1")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="radar_cnn")
    return model


def compile_model(model: tf.keras.Model,
                  learning_rate: float = 1e-3) -> tf.keras.Model:
    """Attach optimiser, loss and metrics to ``model``."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
