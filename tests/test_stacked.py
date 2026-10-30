"""Integration test for the stacked trainer on fully synthetic data."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from name_that_llm.data import LABELS
from name_that_llm.stacked import align_features, train_stacked

LABEL_OFFSETS = {label: index for index, label in enumerate(LABELS)}


@pytest.fixture
def synthetic_features(tmp_path, training_frame):
    rng = np.random.default_rng(7)
    rows = []
    for _, row in training_frame.iterrows():
        shift = 10.0 * LABEL_OFFSETS[row["model"]]
        rows.append({"id": row["id"], **{f"f{i}": rng.normal(shift, 1.0) for i in range(4)}})
    path = tmp_path / "features.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_align_features_reorders_and_rejects_duplicates(tmp_path, training_frame):
    scrambled = training_frame.sample(frac=1.0, random_state=0)["id"]
    table = pd.DataFrame({"id": training_frame["id"], "f0": np.arange(len(training_frame))})
    path = tmp_path / "f.parquet"
    table.to_parquet(path, index=False)
    aligned = align_features(path, scrambled)
    expected = table.set_index("id").loc[scrambled, "f0"]
    assert aligned["f0"].tolist() == expected.tolist()

    duplicated = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    duplicated.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Duplicate feature ids"):
        align_features(path, training_frame["id"])


def test_stacked_pipeline_writes_bundle_and_metrics(tmp_path, data_dir, synthetic_features):
    output_dir = tmp_path / "stacked"
    report = train_stacked(data_dir, synthetic_features, output_dir, n_splits=3, seed=42)
    assert 0 <= report["oof_balanced_accuracy"] <= 1
    assert len(report["folds"]) == 3
    assert report["config"]["backbone_columns"] == 4
    assert report["config"]["seed"] == 42
    assert report["versions"]["scipy"] != "unknown"

    bundle = joblib.load(output_dir / "model.joblib")
    assert {"vectorizer", "scaler", "classifier"} <= set(bundle)
    assert bundle["backbone_features_sha256"] == report["features_sha256"]

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["oof_balanced_accuracy"] == report["oof_balanced_accuracy"]
    oof = pd.read_csv(output_dir / "oof_predictions.csv", dtype={"prompt_id": str})
    assert oof.groupby("prompt_id")["fold"].nunique().eq(1).all()

    with pytest.raises(ValueError, match="not empty"):
        train_stacked(data_dir, synthetic_features, output_dir)
