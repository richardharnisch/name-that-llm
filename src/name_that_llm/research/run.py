"""Run the first prompt-effect experiment, with exact oracle pairs and triples."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from name_that_llm.data import LABELS, load_folds
from name_that_llm.research.evaluation import summarize
from name_that_llm.research.features import load_embeddings, make_features
from name_that_llm.research.gaussian import SharedPromptGaussian
from name_that_llm.research.groups import (
    center_groups,
    fragment_prompts,
    group_numbers,
    shuffle_groups,
    validate_design,
)


def fit_logistic(x, y: np.ndarray, prompts: np.ndarray, seed: int) -> tuple:
    """Choose C on a training-only prompt holdout, then refit all outer training rows."""
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train, validation = next(split.split(x, y, groups=prompts))
    if set(y[train]) != set(y) or set(y[validation]) != set(y):
        raise ValueError("Logistic tuning split lacks labels; use more training prompts.")
    scores = {}
    for c in (0.1, 1.0, 10.0):
        classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
        classifier.fit(x[train], y[train])
        scores[c] = float(balanced_accuracy_score(y[validation], classifier.predict(x[validation])))
    best = max(scores, key=scores.get)
    classifier = LogisticRegression(C=best, class_weight="balanced", max_iter=2000).fit(x, y)
    return classifier, {"C": best, "training_holdout_scores": scores}


def lda(x: np.ndarray, y: np.ndarray) -> LinearDiscriminantAnalysis:
    return LinearDiscriminantAnalysis(
        solver="lsqr", shrinkage="auto", priors=np.full(len(LABELS), 1 / len(LABELS))
    ).fit(x, y)


def record(
    validation: pd.DataFrame,
    probabilities: np.ndarray,
    method: str,
    seed: int = -1,
    groups: list | None = None,
) -> pd.DataFrame:
    result = validation[["id", "prompt_id", "model", "split", "fold"]].copy()
    true = np.array([LABELS.index(label) for label in result["model"]])
    predicted = probabilities.argmax(axis=1)
    result["prediction"] = np.array(LABELS)[predicted]
    result["correct"] = predicted == true
    result["nll"] = -np.log(np.maximum(probabilities[np.arange(len(true)), true], 1e-300))
    result["method"] = method
    result["partition_seed"] = seed
    result["group"] = (
        group_numbers(groups, len(result)) if groups is not None else np.arange(len(result))
    )
    return result


def run_fold(frame: pd.DataFrame, fold: int, args: argparse.Namespace, embeddings) -> tuple:
    train = frame[frame["fold"] != fold]
    validation = frame[frame["fold"] == fold]
    supervised_predictions, artifact_manifest = None, None
    if args.fold_artifacts:
        from name_that_llm.research.fold_embeddings import load_fold_artifact

        embeddings, supervised_predictions, artifact_manifest = load_fold_artifact(
            args.fold_artifacts / f"fold-{fold}",
            frame,
            fold,
            folds_path=args.data_dir / "folds.csv",
        )
    x, xv, full_x, full_xv = make_features(
        train, validation, args.features, args.dimension, args.max_features, args.seed, embeddings
    )
    y = train["model"].to_numpy()
    prompts = train["prompt_id"].to_numpy()
    validation_prompts = validation["prompt_id"].to_numpy()
    model = SharedPromptGaussian(args.shrinkage, args.max_iter, args.tol).fit(x, y, prompts)
    assert tuple(model.classes_) == LABELS
    info = {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_prompts": train["prompt_id"].nunique(),
        "validation_prompts": validation["prompt_id"].nunique(),
        "em_iterations": model.n_iter_,
        "em_converged": model.converged_,
        "em_relative_change": model.relative_change_,
        "prompt_variance_fraction": float(
            np.trace(model.prompt_cov_) / np.trace(model.prompt_cov_ + model.noise_cov_)
        ),
    }
    predictions = [record(validation, model.predict_proba(xv), "independent")]
    if supervised_predictions is not None:
        predictions.append(supervised_predictions)
        info["supervised_artifact"] = artifact_manifest
    for name, features, val_features in (
        ("logistic_full", full_x, full_xv),
        ("logistic_dense", x, xv),
    ):
        classifier, tuning = fit_logistic(features, y, prompts, args.seed)
        info[name] = tuning
        predictions.append(record(validation, classifier.predict_proba(val_features), name))
    standalone_lda = lda(x, y)
    independent_lda = standalone_lda.predict_proba(xv)
    predictions.append(record(validation, independent_lda, "lda"))
    group_audit = []
    for size in (2, 3):
        for seed in args.partition_seeds:
            groups = fragment_prompts(validation_prompts, size, seed)
            shuffled = shuffle_groups(groups, validation["split"].to_numpy(), seed + 10000)
            for name, batch in ((f"shared_{size}", groups), (f"shuffled_{size}", shuffled)):
                predictions.append(
                    record(validation, model.predict_groups(xv, batch), name, seed, batch)
                )
            # Fit the naive-centering control under the same fragmentation regime.
            # Short singleton remainders use the standalone LDA, not zero-vector guesses.
            training_groups = fragment_prompts(prompts, size, seed)
            centered_train = center_groups(x, training_groups)
            eligible = np.concatenate([group for group in training_groups if len(group) > 1])
            centered_classifier = lda(centered_train[eligible], y[eligible])
            centered_probabilities = centered_classifier.predict_proba(center_groups(xv, groups))
            singletons = np.array([group[0] for group in groups if len(group) == 1], dtype=int)
            centered_probabilities[singletons] = independent_lda[singletons]
            predictions.append(
                record(validation, centered_probabilities, f"centered_{size}", seed, groups)
            )
            pairs = [(i, j) for group in shuffled for i in group for j in group if i < j]
            group_audit.append(
                {
                    "size": size,
                    "seed": seed,
                    "realized_sizes": {str(k): sum(len(g) == k for g in groups) for k in (1, 2, 3)},
                    "shuffled_same_prompt_pair_fraction": float(
                        np.mean([validation_prompts[i] == validation_prompts[j] for i, j in pairs])
                    ),
                }
            )
    info["groups"] = group_audit
    return pd.concat(predictions, ignore_index=True), info


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def provenance(args: argparse.Namespace) -> dict:
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision, status = "unavailable", "unavailable"
    return {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "folds_sha256": digest(args.data_dir / "folds.csv"),
        "embeddings_sha256": digest(args.embeddings) if args.embeddings else None,
        "embedding_provenance": args.embedding_provenance,
        "git_revision": revision,
        "git_status": status,
        "source_sha256": {
            path.name: digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pandas", "scikit-learn", "name-that-llm")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", choices=("word", "char", "embeddings"), default="word")
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--fold-artifacts", type=Path, help="Supervised artifacts under fold-K/")
    parser.add_argument(
        "--embedding-provenance", help="Extractor, revision, pooling, and training history"
    )
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partition-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--fold", type=int, help="Smoke test only; omit for full CV")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.dimension < 1 or args.max_features < 2 or args.bootstrap < 100 or args.threads < 1:
        parser.error("Use positive dimensions/threads and at least 100 bootstrap draws.")
    if len(set(args.partition_seeds)) != len(args.partition_seeds):
        parser.error("Partition seeds must be distinct.")
    if args.fold_artifacts and (args.embeddings or args.features != "embeddings"):
        parser.error("Use --fold-artifacts with --features embeddings and without --embeddings.")
    if (
        args.features == "embeddings"
        and not args.fold_artifacts
        and (not args.embeddings or not args.embedding_provenance)
    ):
        parser.error("Embeddings require --embeddings and --embedding-provenance.")
    if args.features != "embeddings" and args.embeddings:
        parser.error("Use --features embeddings with --embeddings.")
    if args.output.exists():
        parser.error("Output already exists; use a new directory to preserve prior runs.")
    frame = load_folds(args.data_dir).sort_values("id").reset_index(drop=True)
    validate_design(frame)
    folds = sorted(frame["fold"].unique()) if args.fold is None else [args.fold]
    if not set(folds).issubset(frame["fold"].unique()):
        parser.error("Requested fold does not exist.")
    embeddings = load_embeddings(args.embeddings, frame["id"]) if args.embeddings else None
    args.output.mkdir(parents=True)
    manifest = provenance(args)
    manifest["status"] = "running"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.perf_counter()
    predictions, diagnostics = [], {}
    with threadpool_limits(limits=args.threads), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for fold in folds:
            print(f"Fold {fold}: fitting {args.features} features and controls", flush=True)
            rows, info = run_fold(frame, fold, args, embeddings)
            rows.to_parquet(args.output / f"fold-{fold}-predictions.parquet", index=False)
            predictions.append(rows)
            diagnostics[str(fold)] = info
            print(
                f"Fold {fold}: EM {info['em_iterations']} iterations, "
                f"converged={info['em_converged']}",
                flush=True,
            )
    predictions = pd.concat(predictions, ignore_index=True)
    result = summarize(predictions, args.bootstrap, args.seed)
    result["diagnostics"] = diagnostics
    result["elapsed_seconds"] = time.perf_counter() - started
    result["complete_cross_validation"] = args.fold is None
    result["gate_eligible"] = args.fold is None and all(
        info["em_converged"] for info in diagnostics.values()
    )
    result["initial_gate_passed"] = result["gate_eligible"] and all(
        all(criteria.values()) for criteria in result["initial_gate_criteria"].values()
    )
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest["status"] = "complete"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, scores in result["pooled"].items():
        print(f"{name:20s} BA={scores['balanced_accuracy']:.4f} logloss={scores['log_loss']:.4f}")
    print(f"Results: {args.output}; elapsed {result['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
