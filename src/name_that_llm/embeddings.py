"""Encode word tokens with pretrained GloVe or fastText vectors."""

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
from gensim import downloader
from gensim.models import KeyedVectors

from .utils.encoder import Encoder


class WordEmbeddingEncoder(Encoder):
    """Encode raw text as an ordered sequence of word vectors.

    The output has shape (sequence length, embedding width). Pooling and recurrent
    layers consume this sequence separately from tokenization and vector lookup.
    """

    def __init__(
        self,
        vectors: KeyedVectors,
        tokenize_words: Callable[[str], Sequence[str]] = str.split,
    ):
        self.vectors = vectors
        self.tokenize_words = tokenize_words

    def __len__(self) -> int:
        return self.vectors.vector_size

    def __call__(self, x: str) -> np.ndarray:
        return self.from_words(self.tokenize_words(x))

    def from_words(self, words: Sequence[str]) -> np.ndarray:
        """Look up each token in order, using a zero vector for missing words."""
        sequence = np.zeros((len(words), len(self)), dtype=np.float32)
        for row, word in enumerate(words):
            try:
                sequence[row] = self.vectors.get_vector(word)
            except KeyError:
                continue
        return sequence


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


def build_embedding_encoder(
    option: str, tokenize_words: Callable[[str], Sequence[str]] = str.split
) -> WordEmbeddingEncoder:
    """Load the selected pretrained vectors and reuse them across documents."""
    path = download_pretrained(option)
    vectors = KeyedVectors.load_word2vec_format(str(path))
    return WordEmbeddingEncoder(vectors, tokenize_words)


def encode_embeddings(
    documents: Iterable[Sequence[str]], encoder: WordEmbeddingEncoder
) -> list[np.ndarray]:
    """Return one dense word-vector matrix per document, in document order."""
    return [encoder.from_words(words) for words in documents]


def encode_mean_pooled(
    texts: Sequence[str], encoder: WordEmbeddingEncoder, *, normalize: bool = False
) -> np.ndarray:
    """Average the word vectors in each text into one fixed-width document vector."""
    features = np.zeros((len(texts), len(encoder)), dtype=np.float32)
    for row, text in enumerate(texts):
        vectors = encoder(text)
        if len(vectors):
            features[row] = vectors.mean(axis=0)
    if normalize:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        np.divide(features, norms, out=features, where=norms > 0)
    return features
