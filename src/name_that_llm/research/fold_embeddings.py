"""Validate supervised artifacts before held-out evaluation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from name_that_llm.data import LABELS
from name_that_llm.research.features import load_embeddings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids(value: object, name: str) -> set[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of string IDs.")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicate IDs.")
    return set(value)


def load_fold_artifact(
    directory: Path, frame: pd.DataFrame, fold: int, *, folds_path: Path
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Require completed training on exactly the requested fold's training IDs."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Fold manifest must be an object.")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("status") != "complete"
    ):
        raise ValueError("Fold manifest requires schema_version 1 and status complete.")
    if type(manifest.get("fold")) is not int or manifest["fold"] != fold:
        raise ValueError("Artifact fold does not match the requested fold.")
    if manifest.get("folds_sha256") != _sha256(folds_path):
        raise ValueError("Artifact folds_sha256 does not match the folds file.")
    if manifest.get("completed_training") is not True:
        raise ValueError("Artifact completed_training must be true.")
    requested, completed = manifest.get("epochs_requested"), manifest.get("epochs_completed")
    if type(requested) is not int or requested < 1 or type(completed) is not int:
        raise ValueError("Training epochs must be positive integers.")
    if completed != requested:
        raise ValueError("Artifact must complete all requested training epochs.")
    for field in ("model", "revision"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"Artifact {field} must be a nonempty string.")
    if not {"id", "fold", "model"}.issubset(frame.columns):
        raise ValueError("Dataset requires id, fold, and model columns.")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("Dataset IDs must be present and unique.")
    if not frame["id"].map(lambda item: isinstance(item, str)).all():
        raise ValueError("Dataset IDs must be strings.")
    if frame["fold"].isna().any() or not set(frame["model"]).issubset(LABELS):
        raise ValueError("Dataset fold or model labels are invalid.")
    validation = frame.loc[frame["fold"] == fold]
    training = frame.loc[frame["fold"] != fold]
    if validation.empty or training.empty:
        raise ValueError("Requested fold requires nonempty training and validation sets.")
    train_ids = _ids(manifest.get("train_ids"), "train_ids")
    validation_ids = _ids(manifest.get("validation_ids"), "validation_ids")
    if train_ids & validation_ids:
        raise ValueError("Training and validation IDs overlap.")
    if train_ids != set(training["id"]) or validation_ids != set(validation["id"]):
        raise ValueError("Artifact training/validation IDs do not match the dataset fold.")
    if "prompt_id" in frame and set(training["prompt_id"]) & set(validation["prompt_id"]):
        raise ValueError("Training and validation prompt IDs overlap.")
    embeddings_path = directory / "embeddings.npz"
    predictions_path = directory / "classifier_predictions.parquet"
    for path, field in (
        (embeddings_path, "embeddings_sha256"),
        (predictions_path, "classifier_predictions_sha256"),
    ):
        if manifest.get(field) != _sha256(path):
            raise ValueError(f"Artifact {field} does not match {path.name}.")
    embeddings = load_embeddings(embeddings_path, frame["id"])
    predictions = pd.read_parquet(predictions_path)
    required = {"id", "model", "fold", "prediction", "nll", "correct", "method"}
    if not required.issubset(predictions.columns):
        raise ValueError("Classifier predictions are missing required columns.")
    if predictions["id"].duplicated().any() or set(predictions["id"]) != validation_ids:
        raise ValueError("Classifier prediction IDs must match validation IDs without duplicates.")
    predictions = predictions.set_index("id").loc[validation["id"]].reset_index()
    for column in ("prompt_id", "split"):
        if column not in validation:
            raise ValueError(f"Dataset requires {column} for classifier evaluation.")
        expected = validation[column].to_numpy()
        if column in predictions and not predictions[column].eq(expected).all():
            raise ValueError(f"Classifier prediction {column} does not match dataset metadata.")
        predictions[column] = expected
    if not predictions["fold"].eq(fold).all():
        raise ValueError("Classifier prediction fold does not match requested fold.")
    if not predictions["model"].eq(validation["model"].to_numpy()).all():
        raise ValueError("Classifier prediction truth labels do not match dataset labels.")
    if not predictions["prediction"].isin(LABELS).all():
        raise ValueError("Classifier predictions contain unknown labels.")
    if not predictions["method"].eq("finetuned_head").all():
        raise ValueError("Classifier prediction method must be finetuned_head.")
    if not pd.api.types.is_numeric_dtype(predictions["nll"]):
        raise ValueError("Classifier prediction nll must be numeric.")
    nll = predictions["nll"].to_numpy(dtype=float)
    if not np.isfinite(nll).all() or (nll < 0).any():
        raise ValueError("Classifier prediction nll must be finite and nonnegative.")
    correct = predictions["prediction"].eq(predictions["model"])
    if predictions["correct"].isna().any() or not predictions["correct"].eq(correct).all():
        raise ValueError("Classifier prediction correct disagrees with predicted labels.")
    manifest = {**manifest, "manifest_sha256": _sha256(directory / "manifest.json")}
    return embeddings, predictions, manifest
