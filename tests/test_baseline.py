import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from name_that_llm.baseline import grouped_folds, train_baseline
from name_that_llm.data import LABELS

ROOT = Path(__file__).resolve().parents[1]


def test_folds_keep_prompts_together_and_cover_every_row(training_frame):
    folds = grouped_folds(training_frame, n_splits=3, seed=42)
    repeat = grouped_folds(training_frame, n_splits=3, seed=42)
    seen = []
    for (train, validation), (again_train, again_validation) in zip(folds, repeat, strict=True):
        assert set(training_frame.iloc[train]["prompt_id"]).isdisjoint(
            training_frame.iloc[validation]["prompt_id"]
        )
        np.testing.assert_array_equal(train, again_train)
        np.testing.assert_array_equal(validation, again_validation)
        seen.extend(validation)
    assert sorted(seen) == list(range(len(training_frame)))


@pytest.mark.parametrize("folds", [1, 7])
def test_rejects_invalid_fold_counts(training_frame, folds):
    with pytest.raises(ValueError, match="Folds must be"):
        grouped_folds(training_frame, folds, seed=42)


def test_rejects_fold_without_all_labels(training_frame):
    incomplete = training_frame.loc[
        ~((training_frame["model"] == LABELS[0]) & (training_frame["prompt_id"] != "prompt000"))
    ]
    with pytest.raises(ValueError, match="missing labels"):
        grouped_folds(incomplete, 3, seed=42)


def run_script(script, *arguments):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_actual_training_and_prediction_scripts(data_dir, training_frame, tmp_path):
    output_dir = tmp_path / "baseline"
    result = run_script(
        "train_baseline.py",
        "--data-dir",
        data_dir,
        "--output-dir",
        output_dir,
        "--folds",
        3,
        "--max-features",
        1000,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output_dir / "metrics.json").read_text())
    assert report["rows"] == len(training_frame)
    assert 0 <= report["oof_balanced_accuracy"] <= 1
    assert len(report["folds"]) == 3
    assert len(report["data_sha256"]) == 64
    oof = pd.read_csv(output_dir / "oof_predictions.csv")
    assert len(oof) == len(training_frame)
    assert oof.groupby("prompt_id")["fold"].nunique().eq(1).all()
    assert not oof["prediction"].isna().any()

    model_path = output_dir / "model.joblib"
    model = joblib.load(model_path)
    assert set(model.classes_) == set(LABELS)
    test_csv = tmp_path / "test.csv"
    test_text = training_frame.iloc[:2]["text"].tolist()
    pd.DataFrame({"id": ["00123", "NA"], "text": test_text}).to_csv(test_csv, index=False)
    output = tmp_path / "submission.parquet"
    result = run_script(
        "predict.py", "--model", model_path, "--test-csv", test_csv, "--output", output
    )
    assert result.returncode == 0, result.stdout + result.stderr
    submission = pd.read_parquet(output)
    assert list(submission.columns) == ["id", "model"]
    assert submission["id"].tolist() == ["00123", "NA"]
    assert submission["model"].tolist() == model.predict(test_text).tolist()

    # Existing results must never be silently replaced.
    result = run_script("predict.py", "--output", output)
    assert result.returncode != 0
    assert "already exists" in result.stderr
    with pytest.raises(ValueError, match="not empty"):
        train_baseline(data_dir, output_dir)


def test_rejects_csv_submission_before_loading_model(tmp_path):
    result = run_script("predict.py", "--output", tmp_path / "submission.csv")
    assert result.returncode != 0
    assert "requires a .parquet" in result.stderr
