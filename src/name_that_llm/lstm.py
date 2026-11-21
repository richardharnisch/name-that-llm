"""A small LSTM that turns a word-vector sequence into a document vector."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence

from name_that_llm.config import DEFAULT_LSTM_HIDDEN_SIZE


class LSTM(nn.Module):
    def __init__(
        self, d_in: int, hidden_size: int = DEFAULT_LSTM_HIDDEN_SIZE, *, normalize: bool = True
    ):
        super().__init__()
        self.recurrent = nn.LSTM(d_in, hidden_size)
        self.normalize = normalize

    def forward(self, sequences: PackedSequence) -> torch.Tensor:
        """Return the final hidden state for each document in the batch."""
        _, (hidden, _) = self.recurrent(sequences)
        features = hidden[-1]
        return F.normalize(features, dim=1) if self.normalize else features
