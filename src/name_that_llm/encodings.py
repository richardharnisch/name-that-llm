from abc import ABC, abstractmethod
from collections import Counter

import numpy as np

from .tokenizer import Tokenizer


class Encoder(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        pass

    @abstractmethod
    def __call__(self, x: str) -> np.ndarray:
        """Represent x as a D-dimensional vector of real numbers"""
        pass


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
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> "np.array":
        token_id_list = list(self.tokenizer.encode(x).ids)

        vector = np.zeros(len(self))
        np.add.at(vector, token_id_list, 1)

        return vector


class TFIDF(Encoder):
    def __init__(self, tokenizer: Tokenizer, documents: np.ndarray):
        self.tokenizer = tokenizer
        N = len(documents)
        df = np.zeros(vocab_size)

        for document in documents:
            token_ids = set(tokenizer.encode(document).ids)
            df[list(token_ids)] += 1

        idf = np.log(N / np.maximum(df, 1))

    def __len__(self) -> int:
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> "np.array":
        token_id_list = list(self.tokenizer.encode(x).ids)

        tf = np.zeros(len(self))
        if len(token_id_list) > 0:
            tf /= len(token_id_list)
        np.add.at(tf, token_id_list, 1)

        return tf * self.idf
