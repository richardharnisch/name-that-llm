"""Download pretrained word vectors from the Gensim catalog."""

from pathlib import Path

from gensim import downloader

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
