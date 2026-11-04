from abc import ABC, abstractmethod
from collections.abc import Sequence

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


class L2Normalized(Encoder):
    """Scale the vectors of another encoder to unit L2 norm.

    This has no fitted parameters, so it is safe across folds. A zero vector stays zero.
    """

    def __init__(self, encoder: Encoder):
        self.encoder = encoder

    def __len__(self) -> int:
        return len(self.encoder)

    def __call__(self, x: str) -> np.ndarray:
        return self._scale(self.encoder(x))

    def from_ids(self, token_ids: Sequence[int]) -> np.ndarray:
        """Represent saved token IDs without tokenizing the text again."""
        return self._scale(self.encoder.from_ids(token_ids))

    @staticmethod
    def _scale(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector


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
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> np.ndarray:
        """Represent x as a D-dimensional vector of real numbers"""
        return self.from_ids(self.tokenizer.encode(x).ids)

    def from_ids(self, token_ids: Sequence[int]) -> np.ndarray:
        """Represent saved token IDs without tokenizing the text again."""
        vector = np.zeros(len(self))
        np.add.at(vector, list(token_ids), 1)

        return vector


class TFIDF(Encoder):
    def __init__(self, tokenizer: Tokenizer, documents: np.ndarray):
        self.tokenizer = tokenizer
        N = len(documents)
        if N == 0:
            raise ValueError("TF-IDF requires at least one training document.")
        df = np.zeros(len(self))

        for document in documents:
            token_ids = set(tokenizer.encode(document).ids)
            df[list(token_ids)] += 1

        self.idf = np.log(N / np.maximum(df, 1))

    @classmethod
    def from_idf(cls, tokenizer: Tokenizer, idf: np.ndarray) -> "TFIDF":
        """Rebuild an encoder from saved IDF weights, without the training texts."""
        idf = np.asarray(idf, dtype=np.float64)
        encoder = cls.__new__(cls)
        encoder.tokenizer = tokenizer
        if idf.shape != (len(encoder),):
            raise ValueError(f"Expected {len(encoder)} IDF weights, got shape {idf.shape}.")
        encoder.idf = idf
        return encoder

    def __len__(self) -> int:
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> np.ndarray:
        return self.from_ids(self.tokenizer.encode(x).ids)

    def from_ids(self, token_ids: Sequence[int]) -> np.ndarray:
        """Represent saved token IDs without tokenizing the text again."""
        token_id_list = list(token_ids)
        tf = np.zeros(len(self))
        np.add.at(tf, token_id_list, 1)
        if len(token_id_list) > 0:
            tf /= len(token_id_list)

        return tf * self.idf
