"""
baselines.py
============
Supervised baselines against which the unsupervised anomaly detector is
benchmarked.

  BinaryCNN     : 2-class fall vs. non-fall classifier
  MulticlassCNN : 6-class activity classifier (falls = argmax == 5)

Both share the encoder backbone with the VAE so that comparisons isolate
the effect of the training objective rather than confounding it with
architectural differences.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from model import _conv_block


class _Backbone(nn.Module):
    def __init__(self, in_channels=1, base=32):
        super().__init__()
        self.b1 = _conv_block(in_channels, base)
        self.b2 = _conv_block(base, base * 2)
        self.b3 = _conv_block(base * 2, base * 4)
        self.b4 = _conv_block(base * 4, base * 8)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        h = self.b4(self.b3(self.b2(self.b1(x))))
        return self.gap(h).flatten(1)


class BinaryCNN(nn.Module):
    """Fall vs not-fall — directly trained against the fall label."""
    def __init__(self, in_channels=1, base=32, dropout=0.3):
        super().__init__()
        self.backbone = _Backbone(in_channels, base)
        self.head = nn.Sequential(
            nn.Linear(base * 8, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class MulticlassCNN(nn.Module):
    """Six-way activity classifier. Output classes are 0..5 corresponding
    to activity codes 1..6 (so subtract 1 from the activity_code label
    before training)."""
    def __init__(self, in_channels=1, base=32, n_classes=6, dropout=0.3):
        super().__init__()
        self.backbone = _Backbone(in_channels, base)
        self.head = nn.Sequential(
            nn.Linear(base * 8, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))
