"""Label-blind batch construction and split checks."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def validate_design(frame: pd.DataFrame) -> None:
    if frame.groupby("prompt_id")["fold"].nunique().max() != 1:
        raise ValueError("A prompt occurs in multiple folds.")
    if frame.groupby("prompt_id")["split"].nunique().max() != 1:
        raise ValueError("A prompt occurs in multiple sources; audit its identity first.")
    normalized = frame["text"].str.split().str.join(" ")
    if normalized.duplicated().any():
        raise ValueError("Duplicate text requires an audit before evaluation.")
    labels = set(frame["model"])
    for fold in frame["fold"].unique():
        for mask in (frame["fold"] == fold, frame["fold"] != fold):
            if set(frame.loc[mask, "model"]) != labels:
                raise ValueError("Every training and validation fold needs all labels.")


def fragment_prompts(prompts: NDArray, size: int, seed: int) -> list[NDArray]:
    """Partition each prompt without inspecting labels; retain short remainders."""
    if size not in (1, 2, 3):
        raise ValueError("Group size must be 1, 2, or 3.")
    rng = np.random.default_rng(seed)
    groups = []
    for prompt in np.unique(prompts):
        indices = rng.permutation(np.flatnonzero(prompts == prompt))
        groups.extend(indices[start : start + size] for start in range(0, len(indices), size))
    return groups


def shuffle_groups(groups: list[NDArray], sources: NDArray, seed: int) -> list[NDArray]:
    """Randomize membership within source; keep group sizes and source composition."""
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(sources))
    for source in np.unique(sources):
        indices = np.flatnonzero(sources == source)
        permutation[indices] = rng.permutation(indices)
    return [permutation[group] for group in groups]


def center_groups(x: NDArray, groups: list[NDArray]) -> NDArray:
    """Subtract observed group means; singleton rows become zero."""
    result = np.empty_like(x)
    for group in groups:
        result[group] = x[group] - x[group].mean(axis=0)
    return result


def group_numbers(groups: list[NDArray], n: int) -> NDArray:
    numbers = np.empty(n, dtype=int)
    for number, group in enumerate(groups):
        numbers[group] = number
    return numbers
