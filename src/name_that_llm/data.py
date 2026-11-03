"""Dataset checks shared by training and prediction."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

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
DEFAULT_VAL_SIZE = DEFAULT_SPLIT.val_size
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
    """Load eligible source rows before the train/validation split."""
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


def split_training_data(
    frame: pd.DataFrame,
    val_size: float = DEFAULT_VAL_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split whole prompts, with stable membership and row order for the same input."""
    validate_columns(frame, ("id", "prompt_id", "text", "model", "split"))
    if not 0 < val_size < 1:
        raise ValueError("Validation size must be between 0 and 1 (exclusive).")
    if frame["prompt_id"].nunique() < 2:
        raise ValueError(
            "At least two distinct prompts are needed for train/validation splitting."
        )
    # A fixed seed alone does not guarantee stable CSV row order.
    frame = frame.sort_values("id").reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_indices, val_indices = next(splitter.split(frame, groups=frame["prompt_id"]))
    train = frame.iloc[train_indices].reset_index(drop=True)
    val = frame.iloc[val_indices].reset_index(drop=True)
    labels = set(frame["model"])
    if set(train["model"]) != labels or set(val["model"]) != labels:
        raise ValueError(
            "Both train and validation must contain every model. "
            "Use more prompts or change --val-size or --seed."
        )
    return train, val


def write_data_splits(
    directory: Path,
    output: Path,
    val_size: float = DEFAULT_VAL_SIZE,
    seed: int = DEFAULT_SEED,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write train.csv and val.csv without changing the downloaded source files."""
    config = SplitConfig(splits=splits, val_size=val_size, seed=seed)
    train, val = split_training_data(
        load_training_data(directory, splits), val_size, seed
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train), ("val", val)):
        subset.to_csv(
            output / f"{name}.csv", index=False, encoding="utf-8", lineterminator="\n"
        )
    save_config(output / "split_config.json", split=config)
    return train, val


def load_test_data(path: Path) -> pd.DataFrame:
    # String dtype preserves IDs such as 00123. Do not treat 'NA' as a missing ID.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    validate_columns(frame, ("id", "text"))
    return frame.loc[:, ["id", "text"]]
