"""Train the TF-IDF + frozen-backbone stacked classifier (iteration 2).

Validates with the same prompt-grouped folds as the TF-IDF baseline, then refits
on all selected passages and saves a reusable inference bundle.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import sklearn
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from name_that_llm.baseline import DEFAULT_SPLITS, grouped_folds, make_text_vectorizer
from name_that_llm.data import load_training_data
from name_that_llm.download import DEFAULT_DATA_DIR, sha256
from name_that_llm.lm_features import DEFAULT_LAYERS, DEFAULT_MAX_LENGTH

DEFAULT_FEATURES = Path("artifacts/features/lm_v2.parquet")
NON_FEATURE_COLUMNS = ("id", "row_index")


def align_features(features_path: Path, passage_ids: pd.Series) -> pd.DataFrame:
    """Return backbone columns only, in the exact row order of passage_ids."""
    table = pd.read_parquet(features_path)
    if "id" not in table.columns:
        raise ValueError(f"{features_path} has no id column.")
    duplicates = table["id"][table["id"].duplicated()]
    if not duplicates.empty:
        raise ValueError(f"Duplicate feature ids: {sorted(set(duplicates))[:3]}")
    ordered = table.set_index("id").reindex(passage_ids.to_numpy())
    if ordered.isna().any().any():
        raise ValueError(
            f"Feature rows missing for {int(ordered.isna().any(axis=1).sum())} passages."
        )
    feature_columns = [name for name in ordered.columns if name not in NON_FEATURE_COLUMNS]
    return ordered.reset_index(drop=True)[feature_columns]


def train_stacked(
    data_dir: Path = DEFAULT_DATA_DIR,
    features_path: Path = DEFAULT_FEATURES,
    output_dir: Path = Path("artifacts/stacked-v2"),
    *,
    n_splits: int = 5,
    seed: int = 42,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    c: float = 1.0,
) -> dict:
    if not 0 < c:
        raise ValueError("C must be positive.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"{output_dir} is not empty. Choose a new --output-dir.")
    frame = load_training_data(data_dir, splits)
    numeric_all = align_features(features_path, frame["id"])
    folds = grouped_folds(frame, n_splits, seed)
    predictions = np.empty(len(frame), dtype=object)
    fold_ids = np.full(len(frame), -1, dtype=int)
    scores = []
    print(f"Stacking text vectors with {numeric_all.shape[1]} backbone features.", flush=True)
    print(
        f"Training on {len(frame):,} passages / {frame['prompt_id'].nunique()} prompts.", flush=True
    )
    for fold, (train, validation) in enumerate(folds):
        vectorizer = make_text_vectorizer()
        scaler = StandardScaler()
        classifier = LinearSVC(C=c, class_weight="balanced", random_state=seed, max_iter=5000)
        text_train = vectorizer.fit_transform(frame.iloc[train]["text"])
        text_validation = vectorizer.transform(frame.iloc[validation]["text"])
        numeric_train = scaler.fit_transform(numeric_all.iloc[train])
        numeric_validation = scaler.transform(numeric_all.iloc[validation])
        x_train = sp.hstack([text_train, sp.csr_matrix(numeric_train)], format="csr")
        x_validation = sp.hstack([text_validation, sp.csr_matrix(numeric_validation)], format="csr")
        classifier.fit(x_train, frame.iloc[train]["model"])
        predictions[validation] = classifier.predict(x_validation)
        fold_ids[validation] = fold
        score = float(
            balanced_accuracy_score(frame.iloc[validation]["model"], predictions[validation])
        )
        scores.append(
            {
                "fold": fold,
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "balanced_accuracy": score,
            }
        )
        print(f"Fold {fold + 1}/{n_splits}: balanced accuracy = {score:.4f}", flush=True)

    model_meta = {
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
        "config": {
            "seed": seed,
            "n_splits": n_splits,
            "splits": list(splits),
            "C": c,
            "backbone_features": str(features_path),
            "backbone_columns": int(numeric_all.shape[1]),
            "backbone_layers": DEFAULT_LAYERS,
            "backbone_max_length": DEFAULT_MAX_LENGTH,
        },
        "data_file": str(data_dir / "passages_clean.csv"),
        "data_sha256": sha256(data_dir / "passages_clean.csv"),
        "features_sha256": sha256(features_path),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
    }

    print("Refitting the final stacked model on all selected passages...", flush=True)
    vectorizer = make_text_vectorizer()
    scaler = StandardScaler()
    classifier = LinearSVC(C=c, class_weight="balanced", random_state=seed, max_iter=5000)
    text_all = vectorizer.fit_transform(frame["text"])
    numeric_scaled = scaler.fit_transform(numeric_all)
    classifier.fit(
        sp.hstack([text_all, sp.csr_matrix(numeric_scaled)], format="csr"), frame["model"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "scaler": scaler,
            "classifier": classifier,
            "backbone_features_sha256": sha256(features_path),
            "data_sha256": sha256(data_dir / "passages_clean.csv"),
        },
        output_dir / "model.joblib",
        compress=3,
    )
    frame.loc[:, ["id", "prompt_id", "model", "split"]].assign(
        prediction=predictions, fold=fold_ids
    ).to_csv(output_dir / "oof_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(model_meta, indent=2) + "\n")
    print(f"OOF balanced accuracy: {model_meta['oof_balanced_accuracy']:.4f}")
    print(f"Saved stacked model bundle, metrics, and OOF predictions to {output_dir}")
    return model_meta


def baseline_oof_agreement(output_dir: Path, previous_dir: Path) -> float | None:
    """Share of identical OOF predictions versus the previous model, same folds."""
    previous_oof = Path(previous_dir) / "oof_predictions.csv"
    if not previous_oof.is_file():
        return None
    current = pd.read_csv(output_dir / "oof_predictions.csv", dtype=str)
    previous = pd.read_csv(previous_oof, dtype=str)
    if not current[["id", "model"]].equals(previous[["id", "model"]]):
        return None
    return float((current["prediction"] == previous["prediction"]).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stacked-v2"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", nargs="+", choices=DEFAULT_SPLITS, default=list(DEFAULT_SPLITS))
    parser.add_argument("--c", type=float, default=1.0)
    args = parser.parse_args()
    train_stacked(
        args.data_dir,
        args.features,
        args.output_dir,
        n_splits=args.folds,
        seed=args.seed,
        splits=tuple(args.splits),
        c=args.c,
    )
    agreement = baseline_oof_agreement(args.output_dir, Path("artifacts/baseline"))
    if agreement is not None:
        print(f"OOF prediction agreement with baseline: {agreement:.4f}")


if __name__ == "__main__":
    main()
