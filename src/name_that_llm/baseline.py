"""Train and run a word/character TF-IDF + linear SVM baseline on CPU."""

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from name_that_llm.data import (
    DEFAULT_SPLITS,
    LABELS,
    load_test_data,
    load_training_data,
    validate_columns,
)
from name_that_llm.download import DEFAULT_DATA_DIR, sha256


def build_model(
    *, seed: int = 42, min_df: int = 2, max_features: int = 100_000, c: float = 1.0
) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_features=max_features,
                    sublinear_tf=True,
                    lowercase=False,
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=min_df,
                    max_features=max_features,
                    sublinear_tf=True,
                    lowercase=False,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LinearSVC(C=c, class_weight="balanced", random_state=seed, max_iter=5000),
            ),
        ]
    )


def grouped_folds(frame: pd.DataFrame, n_splits: int, seed: int) -> list:
    if not 2 <= n_splits <= frame["prompt_id"].nunique():
        raise ValueError("Folds must be between 2 and the number of distinct prompts.")
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(frame, frame["model"], groups=frame["prompt_id"]))
    for train, validation in folds:
        for indices in (train, validation):
            if set(frame.iloc[indices]["model"]) != set(LABELS):
                raise ValueError("A fold is missing labels. Use fewer folds or more prompt groups.")
    return folds


def train_baseline(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = Path("artifacts/baseline"),
    *,
    n_splits: int = 5,
    seed: int = 42,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    min_df: int = 2,
    max_features: int = 100_000,
    c: float = 1.0,
) -> dict:
    if min_df < 1 or max_features < 1 or c <= 0:
        raise ValueError("min_df, max_features, and C must be positive.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"{output_dir} is not empty. Choose a new --output-dir to preserve it.")
    frame = load_training_data(data_dir, splits)
    folds = grouped_folds(frame, n_splits, seed)
    model = build_model(seed=seed, min_df=min_df, max_features=max_features, c=c)
    predictions = np.empty(len(frame), dtype=object)
    fold_ids = np.full(len(frame), -1, dtype=int)
    scores = []
    print(
        f"Training on {len(frame):,} passages / {frame['prompt_id'].nunique()} prompts.", flush=True
    )
    for fold, (train, validation) in enumerate(folds):
        # Fit vocabulary and IDF only on training prompts, not on validation text.
        fitted = clone(model).fit(frame.iloc[train]["text"], frame.iloc[train]["model"])
        predictions[validation] = fitted.predict(frame.iloc[validation]["text"])
        fold_ids[validation] = fold
        score = float(
            balanced_accuracy_score(frame.iloc[validation]["model"], predictions[validation])
        )
        scores.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "balanced_accuracy": score,
            }
        )
        print(f"Fold {fold + 1}/{n_splits}: balanced accuracy = {score:.4f}", flush=True)

    report = {
        "metric": "balanced_accuracy",
        "validation_source": "expanded dataset only; not a Kaggle leaderboard score",
        "oof_balanced_accuracy": float(balanced_accuracy_score(frame["model"], predictions)),
        "fold_mean": float(np.mean([item["balanced_accuracy"] for item in scores])),
        "fold_std": float(np.std([item["balanced_accuracy"] for item in scores])),
        "folds": scores,
        "rows": len(frame),
        "prompt_groups": int(frame["prompt_id"].nunique()),
        "label_counts": {
            label: int(count) for label, count in frame["model"].value_counts().items()
        },
        "classification_report": classification_report(
            frame["model"], predictions, labels=list(LABELS), output_dict=True, zero_division=0
        ),
        "config": {
            "seed": seed,
            "n_splits": n_splits,
            "splits": list(splits),
            "min_df": min_df,
            "max_features_per_view": max_features,
            "C": c,
        },
        "data_file": str(data_dir / "passages_clean.csv"),
        "data_sha256": sha256(data_dir / "passages_clean.csv"),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    # Refit on every selected external passage after validation.
    print("Refitting the final model on all selected passages...", flush=True)
    model.fit(frame["text"], frame["model"])
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib", compress=3)
    frame.loc[:, ["id", "prompt_id", "model", "split"]].assign(
        prediction=predictions, fold=fold_ids
    ).to_csv(output_dir / "oof_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"OOF balanced accuracy: {report['oof_balanced_accuracy']:.4f}")
    print(f"Saved model, metrics, and out-of-fold predictions to {output_dir}")
    return report


def predict_submission(model_path: Path, test_csv: Path, output: Path) -> pd.DataFrame:
    if output.suffix != ".parquet":
        raise ValueError("This competition requires a .parquet submission, not CSV.")
    if output.exists():
        raise ValueError(f"{output} already exists. Choose a new --output.")
    frame = load_test_data(test_csv)
    # joblib is pickle-based: only load models that you trained or otherwise trust.
    model = joblib.load(model_path)
    if set(model.classes_) != set(LABELS):
        raise ValueError("The fitted model does not cover the 18 competition labels.")
    submission = pd.DataFrame({"id": frame["id"], "model": model.predict(frame["text"])})
    validate_columns(submission, ("id", "model"))
    if not set(submission["model"]).issubset(LABELS):
        raise ValueError("Predictions contain unknown labels.")
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(output, index=False)
    print(f"Saved {len(submission):,} predictions to {output}")
    return submission


def train_main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and validate the expanded-data CPU baseline."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", nargs="+", choices=DEFAULT_SPLITS, default=list(DEFAULT_SPLITS))
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=100_000, help="Limit per TF-IDF view")
    parser.add_argument("--c", type=float, default=1.0, help="Linear SVM regularisation parameter")
    args = parser.parse_args()
    try:
        train_baseline(
            args.data_dir,
            args.output_dir,
            n_splits=args.folds,
            seed=args.seed,
            splits=tuple(args.splits),
            min_df=args.min_df,
            max_features=args.max_features,
            c=args.c,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Training failed: {error}\n")


def predict_main() -> None:
    parser = argparse.ArgumentParser(description="Write a Name That LLM submission.parquet.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/baseline/model.joblib"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/raw/competition/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/submission.parquet"))
    args = parser.parse_args()
    try:
        predict_submission(args.model, args.test_csv, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Prediction failed: {error}\n")
