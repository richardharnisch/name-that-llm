"""Reject fold artifacts that would invalidate held-out evaluation."""

import json

import numpy as np
import pandas as pd
import pytest

from name_that_llm.data import LABELS
from name_that_llm.research.fold_embeddings import _sha256, load_fold_artifact


@pytest.fixture
def artifact(tmp_path):
    frame = pd.DataFrame({"id": ["a", "b", "c"], "fold": [0, 1, 0], "model": [LABELS[0]] * 3})
    frame["prompt_id"] = ["p0", "p1", "p2"]
    frame["split"] = "train"
    folds = tmp_path / "folds.csv"
    frame.to_csv(folds, index=False)
    np.savez(tmp_path / "embeddings.npz", ids=["c", "b", "a"], embeddings=[[3, 4], [2, 3], [1, 2]])
    predictions = frame.iloc[[2, 0]].copy()
    predictions["prediction"] = LABELS[0]
    predictions["correct"] = True
    predictions["nll"] = 0.2
    predictions["method"] = "finetuned_head"
    predictions.to_parquet(tmp_path / "classifier_predictions.parquet", index=False)
    manifest = dict(
        schema_version=1,
        status="complete",
        fold=0,
        folds_sha256=_sha256(folds),
        train_ids=["b"],
        validation_ids=["a", "c"],
        model="encoder",
        revision="revision",
        epochs_requested=1,
        epochs_completed=1,
        completed_training=True,
        embeddings_sha256=_sha256(tmp_path / "embeddings.npz"),
        classifier_predictions_sha256=_sha256(tmp_path / "classifier_predictions.parquet"),
    )
    return tmp_path, frame, folds, manifest


def load(artifact):
    directory, frame, folds, manifest = artifact
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return load_fold_artifact(directory, frame, 0, folds_path=folds)


def test_aligns_embeddings_and_predictions(artifact):
    embeddings, predictions, _ = load(artifact)
    np.testing.assert_array_equal(embeddings, [[1, 2], [2, 3], [3, 4]])
    assert predictions["id"].tolist() == ["a", "c"]


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("fold", 1, "Artifact fold"),
        ("folds_sha256", "bad", "folds_sha256"),
        ("train_ids", ["b", "a"], "overlap"),
        ("train_ids", ["b", "b"], "duplicate"),
        ("train_ids", [], "do not match"),
        ("completed_training", False, "completed_training"),
        ("epochs_completed", 0, "complete all"),
        ("embeddings_sha256", "bad", "embeddings_sha256"),
        ("classifier_predictions_sha256", "bad", "classifier_predictions_sha256"),
    ],
)
def test_rejects_invalid_manifest(artifact, key, value, message):
    artifact[3][key] = value
    with pytest.raises(ValueError, match=message):
        load(artifact)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("id", "a", "IDs"),
        ("model", LABELS[1], "truth"),
        ("prediction", "unknown", "unknown"),
        ("correct", False, "correct"),
        ("fold", 1, "fold"),
        ("method", "independent", "method"),
        ("nll", -1, "nonnegative"),
        ("nll", float("inf"), "finite"),
    ],
)
def test_rejects_invalid_predictions(artifact, column, value, message):
    path = artifact[0] / "classifier_predictions.parquet"
    predictions = pd.read_parquet(path)
    predictions[column] = value
    predictions.to_parquet(path, index=False)
    artifact[3]["classifier_predictions_sha256"] = _sha256(path)
    with pytest.raises(ValueError, match=message):
        load(artifact)


def test_rejects_nonfinite_embeddings(artifact):
    path = artifact[0] / "embeddings.npz"
    np.savez(path, ids=["a", "b", "c"], embeddings=[[np.nan], [0], [1]])
    artifact[3]["embeddings_sha256"] = _sha256(path)
    with pytest.raises(ValueError, match="nonfinite"):
        load(artifact)


def test_rejects_prompt_leakage(artifact):
    artifact[1]["prompt_id"] = ["shared", "shared", "other"]
    with pytest.raises(ValueError, match="prompt IDs overlap"):
        load(artifact)


def test_records_manifest_digest(artifact):
    _, _, manifest = load(artifact)
    assert manifest["manifest_sha256"] == _sha256(artifact[0] / "manifest.json")


@pytest.mark.parametrize("column", ["prompt_id", "split"])
def test_rejects_wrong_evaluation_metadata(artifact, column):
    path = artifact[0] / "classifier_predictions.parquet"
    predictions = pd.read_parquet(path)
    predictions[column] = "wrong"
    predictions.to_parquet(path, index=False)
    artifact[3]["classifier_predictions_sha256"] = _sha256(path)
    with pytest.raises(ValueError, match=column):
        load(artifact)


def test_restores_evaluation_metadata_from_dataset(artifact):
    path = artifact[0] / "classifier_predictions.parquet"
    predictions = pd.read_parquet(path).drop(columns=["prompt_id", "split"])
    predictions.to_parquet(path, index=False)
    artifact[3]["classifier_predictions_sha256"] = _sha256(path)
    _, predictions, _ = load(artifact)
    assert predictions["prompt_id"].tolist() == ["p0", "p2"]
    assert predictions["split"].tolist() == ["train", "train"]
