from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .tokenizer import Tokenizer

ENCODERS = ("bag-of-words", "token-indicator", "tf-idf")


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


def build_encoder(
    name: str, tokenizer: Tokenizer, train_texts: pd.Series, normalize: bool = False
) -> Encoder:
    """Create an encoder. TF-IDF weights come from training texts only."""
    if name == "bag-of-words":
        encoder = BagOfWords(tokenizer)
    elif name == "token-indicator":
        encoder = TokenIndicator(tokenizer)
    else:
        encoder = TFIDF(tokenizer, train_texts.to_numpy())
    return L2Normalized(encoder) if normalize else encoder


def encoder_from_checkpoint(checkpoint: dict, tokenizer: Tokenizer) -> Encoder:
    """Rebuild the encoder a saved model was trained with."""
    name = checkpoint["encoder"]
    if name == "tf-idf":
        encoder = TFIDF.from_idf(tokenizer, checkpoint["idf"].numpy())
        return L2Normalized(encoder) if checkpoint.get("normalize", False) else encoder
    return build_encoder(
        name, tokenizer, pd.Series(dtype=str), checkpoint.get("normalize", False)
    )


def encode_sparse(input_ids: pd.Series, encoder: Encoder) -> sp.csr_matrix:
    """Encode saved token IDs into a sparse document-feature matrix.

    Build one dense document vector at a time and retain its nonzero entries.
    """
    indices, data, indptr = [], [], [0]
    for ids in input_ids:
        vector = encoder.from_ids(ids)
        nonzero = np.flatnonzero(vector)
        indices.append(nonzero.astype(np.int32))
        data.append(vector[nonzero].astype(np.float32))
        indptr.append(indptr[-1] + len(nonzero))
    return sp.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), np.asarray(indptr)),
        shape=(len(input_ids), len(encoder)),
    )
