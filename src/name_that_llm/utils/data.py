"""Dataset checks shared by training and prediction."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from name_that_llm.config import DEFAULT_SPLIT, SplitConfig, save_config

LABELS = (
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
)
# Keep the existing imports available; config.py owns the defaults.
DEFAULT_SPLITS = DEFAULT_SPLIT.splits
DEFAULT_N_FOLDS = DEFAULT_SPLIT.n_folds
DEFAULT_SEED = DEFAULT_SPLIT.seed


def validate_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("The input has no rows.")
    for column in columns:
        if (
            frame[column].isna().any()
            or frame[column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"Column {column!r} contains missing or empty values.")
    if "id" in columns and frame["id"].duplicated().any():
        raise ValueError("The input contains duplicate IDs.")


def load_training_data(
    directory: Path, splits: tuple[str, ...] = DEFAULT_SPLITS
) -> pd.DataFrame:
    """Load eligible source rows before the cross-validation split."""
    path = directory / "passages_clean.csv"
    if not path.is_file():
        raise ValueError(
            f"Missing {path}. Run `uv run python scripts/download_data.py` first."
        )
    if not splits or set(splits) - set(DEFAULT_SPLITS):
        raise ValueError(f"Use one or more of these training splits: {DEFAULT_SPLITS}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    columns = ("id", "prompt_id", "text", "model", "split")
    validate_columns(frame, columns)
    frame = frame.loc[frame["split"].isin(splits), list(columns)].reset_index(drop=True)
    validate_columns(frame, columns)
    labels = set(frame["model"])
    if labels != set(LABELS):
        raise ValueError(
            f"Expected all 18 competition labels. Missing: {sorted(set(LABELS) - labels)}; "
            f"unknown: {sorted(labels - set(LABELS))}"
        )
    normalised = frame["text"].str.split().str.join(" ")
    if normalised.duplicated().any():
        raise ValueError(
            "Duplicate passage text could leak across folds. Check the dataset."
        )
    return frame


def assign_folds(
    frame: pd.DataFrame,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Add a `fold` column. Whole prompts stay in one fold, and labels are balanced.

    For fold k, rows with fold == k are validation and all other rows are training.
    Rows are sorted by ID first, so the result does not depend on the CSV row order.
    """
    validate_columns(frame, ("id", "prompt_id", "text", "model", "split"))
    if n_folds < 2:
        raise ValueError("At least two folds are needed.")
    if frame["prompt_id"].nunique() < n_folds:
        raise ValueError(f"At least {n_folds} distinct prompts are needed for {n_folds} folds.")
    frame = frame.sort_values("id").reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.full(len(frame), -1)
    for fold, (_, val_indices) in enumerate(
        splitter.split(frame, frame["model"], groups=frame["prompt_id"])
    ):
        folds[val_indices] = fold
    frame["fold"] = folds
    labels = set(frame["model"])
    for fold in range(n_folds):
        in_fold = frame["fold"] == fold
        if set(frame.loc[in_fold, "model"]) != labels:
            raise ValueError(f"Fold {fold} validation lacks a model. Change --folds or --seed.")
        if set(frame.loc[~in_fold, "model"]) != labels:
            raise ValueError(f"Fold {fold} training lacks a model. Change --folds or --seed.")
    return frame


def fold_summary(frame: pd.DataFrame) -> str:
    counts = frame.groupby("fold").size()
    return f"{len(counts)} folds; rows per fold: {', '.join(f'{n:,}' for n in counts)}"


def split_fold(frame: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (training, validation) rows for one cross-validation fold."""
    if fold not in set(frame["fold"]):
        raise ValueError(f"Fold {fold} does not exist. Folds: {sorted(set(frame['fold']))}.")
    return (
        frame[frame["fold"] != fold].reset_index(drop=True),
        frame[frame["fold"] == fold].reset_index(drop=True),
    )


def load_folds(directory: Path) -> pd.DataFrame:
    """Read folds.csv, the output of write_data_splits."""
    path = directory / "folds.csv"
    if not path.is_file():
        raise ValueError(f"Missing {path}. Run `uv run python scripts/split_data.py` first.")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    validate_columns(frame, ("id", "prompt_id", "text", "model", "split", "fold"))
    frame["fold"] = frame["fold"].astype(int)
    if set(frame["model"]) != set(LABELS):
        raise ValueError(f"{path} must contain all 18 model labels and no unknown labels.")
    return frame


def write_data_splits(
    directory: Path,
    output: Path,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_SEED,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> pd.DataFrame:
    """Write folds.csv without changing the downloaded source files."""
    config = SplitConfig(splits=splits, n_folds=n_folds, seed=seed)
    frame = assign_folds(load_training_data(directory, splits), n_folds, seed)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "folds.csv", index=False, encoding="utf-8", lineterminator="\n")
    save_config(output / "split_config.json", split=config)
    return frame


def load_test_data(path: Path) -> pd.DataFrame:
    # String dtype preserves IDs such as 00123. Do not treat 'NA' as a missing ID.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    validate_columns(frame, ("id", "text"))
    return frame.loc[:, ["id", "text"]]
