"""Train-fold-only feature transforms; no model downloads."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def load_embeddings(path: Path, ids: pd.Series) -> np.ndarray:
    """Require one vector per dataset ID, independent of archive row ordering."""
    with np.load(path, allow_pickle=False) as archive:
        keys = archive["ids"].astype(str)
        embeddings = archive["embeddings"].astype(float)
    if embeddings.ndim != 2 or len(keys) != len(embeddings):
        raise ValueError("Embeddings need a 2D matrix and a matching 1D ids array.")
    if len(set(keys)) != len(keys) or set(keys) != set(ids):
        raise ValueError("Embedding IDs must match dataset IDs exactly, without duplicates.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain nonfinite values.")
    positions = {key: position for position, key in enumerate(keys)}
    return embeddings[[positions[key] for key in ids]]


def make_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    kind: str,
    dimension: int,
    max_features: int,
    seed: int,
    embeddings: np.ndarray | None = None,
) -> tuple:
    """Return dense train/validation and full train/validation baseline features."""
    if embeddings is not None:
        full_train = embeddings[train.index]
        full_validation = embeddings[validation.index]
        if dimension >= min(full_train.shape):
            raise ValueError("Reduce --dimension below training rows and embedding width.")
        reducer = PCA(n_components=dimension, svd_solver="randomized", random_state=seed)
    else:
        vectorizer = TfidfVectorizer(
            analyzer="word" if kind == "word" else "char_wb",
            ngram_range=(1, 2) if kind == "word" else (3, 5),
            min_df=2,
            max_features=max_features,
            sublinear_tf=True,
        )
        full_train = vectorizer.fit_transform(train["text"])
        full_validation = vectorizer.transform(validation["text"])
        if dimension >= min(full_train.shape):
            raise ValueError("Reduce --dimension below training rows and vocabulary size.")
        reducer = TruncatedSVD(n_components=dimension, random_state=seed)
    dense_train = reducer.fit_transform(full_train)
    dense_validation = reducer.transform(full_validation)
    return dense_train, dense_validation, full_train, full_validation
