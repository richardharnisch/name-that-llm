"""Word embedding interface and Gensim downloads."""

from abc import abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from gensim import downloader

from .utils.encoder import Encoder


class WordEmbeddingEncoder(Encoder):
    """Encode raw text as an ordered sequence of word vectors.

    The output has shape (sequence length, embedding width). Pooling and recurrent
    layers consume this sequence separately from tokenization and vector lookup.
    """

    def __init__(self, tokenize_words: Callable[[str], Sequence[str]]):
        self.tokenize_words = tokenize_words

    def __call__(self, x: str) -> np.ndarray:
        return self.from_words(self.tokenize_words(x))

    @abstractmethod
    def from_words(self, words: Sequence[str]) -> np.ndarray:
        """Look up vectors for word tokens in sequence order."""


PRETRAINED_MODELS = {
    "glove": "glove-wiki-gigaword-50",
    "fasttext": "fasttext-wiki-news-subwords-300",
}


def download_pretrained(option: str) -> Path:
    """Return the path to the cached vectors without loading them into memory."""
    try:
        model_name = PRETRAINED_MODELS[option]
    except KeyError as error:
        raise ValueError(f"Unknown embedding option: {option}") from error
    return Path(downloader.load(model_name, return_path=True))
