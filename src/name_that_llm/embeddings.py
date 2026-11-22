"""Encode word tokens with pretrained GloVe or fastText vectors."""

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from gensim import downloader
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
from nltk.tokenize import word_tokenize

from .utils.encoder import Encoder

WORD_TOKENIZER_NAME = "nltk-sentence-word-lowercase"
FASTTEXT_BINARY_NAME = "wiki-news-300d-1M-subword.bin"
FASTTEXT_BINARY_PATH = Path("models/fasttext") / FASTTEXT_BINARY_NAME
FASTTEXT_BINARY_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/vectors-english/" + FASTTEXT_BINARY_NAME + ".zip"
)


def tokenize_embedding_words(text: str, *, preserve_line: bool = False) -> list[str]:
    """Keep original case for tokenization, then lowercase for lookup."""
    return [token.lower() for token in word_tokenize(text, preserve_line=preserve_line)]


class WordEmbeddingEncoder(Encoder):
    """Encode raw text as an ordered sequence of word vectors.

    The output has shape (sequence length, embedding width). Pooling and recurrent
    layers consume this sequence separately from tokenization and vector lookup.
    """

    def __init__(
        self,
        vectors: KeyedVectors,
        tokenize_words: Callable[[str], Sequence[str]] = tokenize_embedding_words,
    ):
        self.vectors = vectors
        self.tokenize_words = tokenize_words

    def __len__(self) -> int:
        return self.vectors.vector_size

    def __call__(self, x: str) -> np.ndarray:
        return self.from_words(self.tokenize_words(x))

    def from_words(self, words: Sequence[str]) -> np.ndarray:
        """Resolve each token in order, using zeros when no vector is available."""
        sequence = np.zeros((len(words), len(self)), dtype=np.float32)
        for row, word in enumerate(words):
            try:
                sequence[row] = self.vectors.get_vector(word)
            except KeyError:
                continue
        return sequence


PRETRAINED_MODELS = {
    "glove": "glove-wiki-gigaword-300",
    "fasttext": FASTTEXT_BINARY_NAME,
}
EMBEDDING_POOLING = ("mean", "lstm")


def download_pretrained(option: str) -> Path:
    """Return the path to the cached vectors without loading them into memory."""
    try:
        model_name = PRETRAINED_MODELS[option]
    except KeyError as error:
        raise ValueError(f"Unknown embedding option: {option}") from error
    if option == "fasttext":
        destination = FASTTEXT_BINARY_PATH
        if not destination.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".fasttext-", dir=destination.parent) as directory:
                archive = Path(directory) / "fasttext.zip"
                payload = Path(directory) / FASTTEXT_BINARY_NAME
                with (
                    urlopen(FASTTEXT_BINARY_URL, timeout=120) as response,
                    archive.open("wb") as out,
                ):
                    copyfileobj(response, out)
                with ZipFile(archive) as zipped:
                    with zipped.open(FASTTEXT_BINARY_NAME) as source, payload.open("wb") as out:
                        copyfileobj(source, out)
                payload.replace(destination)
        return destination
    cached = Path(downloader.BASE_DIR) / model_name / f"{model_name}.gz"
    if cached.is_file():
        return cached
    return Path(downloader.load(model_name, return_path=True))


def build_embedding_encoder(
    option: str, tokenize_words: Callable[[str], Sequence[str]] = tokenize_embedding_words
) -> WordEmbeddingEncoder:
    """Load the selected pretrained vectors and reuse them across documents."""
    path = download_pretrained(option)
    vectors = (
        load_facebook_vectors(str(path))
        if option == "fasttext"
        else KeyedVectors.load_word2vec_format(str(path))
    )
    return WordEmbeddingEncoder(vectors, tokenize_words)


def encode_embeddings(
    documents: Iterable[Sequence[str]], encoder: WordEmbeddingEncoder
) -> list[np.ndarray]:
    """Return one dense word-vector matrix per document, in document order."""
    return [encoder.from_words(words) for words in documents]


def encode_embedding(
    texts: Sequence[str],
    encoder: WordEmbeddingEncoder,
    *,
    pooling: str = "mean",
    normalize: bool = False,
) -> np.ndarray | list[np.ndarray]:
    """Return mean-pooled document vectors or ordered sequences for the LSTM.

    Normalize pooled means here. The LSTM model normalizes its own document vectors.
    """
    if pooling == "lstm":
        return [encoder(text) for text in texts]
    if pooling != "mean":
        raise ValueError(f"Unknown embedding pooling method: {pooling}")

    features = np.zeros((len(texts), len(encoder)), dtype=np.float32)
    for row, text in enumerate(texts):
        vectors = encoder(text)
        if len(vectors):
            features[row] = vectors.mean(axis=0)
    if normalize:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        np.divide(features, norms, out=features, where=norms > 0)
    return features
