from collections.abc import Sequence

import numpy as np

from .tokenizer import Tokenizer


class Encoder:
    def __len__(self) -> int:
        """Dimensionality of the representation space"""

    def __call__(self, x: str) -> np.ndarray:
        """Represent x as a D-dimensional vector of real numbers"""


class TokenIndicator(Encoder):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> np.ndarray:
        """Represent x as a D-dimensional vector of real numbers"""
        return self.from_ids(self.tokenizer.encode(x).ids)

    def from_ids(self, token_ids: Sequence[int]) -> np.ndarray:
        """Represent saved token IDs without tokenizing the text again."""
        unique_token_ids = list(set(token_ids))

        # A 0 vector with the same length as the vocabulary except in positions corresponding
        # to tokens appearing in the document
        vector = np.zeros(len(self))
        vector[unique_token_ids] = 1
        return vector


class BagOfWords(Encoder):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> np.ndarray:
        """Represent x as a D-dimensional vector of real numbers"""
        return self.from_ids(self.tokenizer.encode(x).ids)

    def from_ids(self, token_ids: Sequence[int]) -> np.ndarray:
        """Represent saved token IDs without tokenizing the text again."""
        vector = np.zeros(len(self))
        np.add.at(vector, list(token_ids), 1)

        return vector
