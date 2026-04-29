"""LSTM model for sequence features derived from spectrogram frames."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Input: (N, T, F)
    Output: (N, C) logits
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be > 0")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.num_classes = int(num_classes)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        out_dim = self.hidden_size * (2 if self.bidirectional else 1)
        self.head = nn.Linear(out_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x)
        if self.bidirectional:
            # Last layer forward and backward states.
            h_last = torch.cat([hn[-2], hn[-1]], dim=1)
        else:
            h_last = hn[-1]
        return self.head(h_last)


class LSTMBinaryClassifier(LSTMClassifier):
    """Binary fall vs non-fall; same API as before (C=2)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            num_classes=2,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
