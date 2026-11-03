from collections import Counter

import numpy as np

from .tokenizer import Tokenizer


class Encoder:
    def __len__(self) -> int:
        """Dimensionality of the representation space"""

    def __call__(self, x: str) -> "np.array":
        """Represent x as a D-dimensional vector of real numbers"""


class TokenIndicator(Encoder):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> "np.array":
        """Represent x as a D-dimensional vector of real numbers"""
        unique_token_ids = list(set(self.tokenizer.encode(x).ids))

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

    def __call__(self, x: str) -> "np.array":
        """Represent x as a D-dimensional vector of real numbers"""
        token_id_list = list(self.tokenizer.encode(x).ids)

        vector = np.zeros(len(self))
        np.add.at(vector, token_id_list, 1)

        return vector
