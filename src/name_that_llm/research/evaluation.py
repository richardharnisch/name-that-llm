"""Paired prompt bootstrap; random partitions are repetitions, not new data."""

import numpy as np
import pandas as pd


def metrics(predictions: pd.DataFrame) -> dict:
    # Average repeated group partitions before aggregating. Never ensemble their
    # probabilities: that would give the grouped method an extra test-time budget.
    rows = (
        predictions.groupby(["id", "model"], sort=True)
        .agg(correct=("correct", "mean"), nll=("nll", "mean"))
        .reset_index()
    )
    recalls = rows.groupby("model")["correct"].mean()
    return {
        "balanced_accuracy": float(recalls.mean()),
        "accuracy": float(rows["correct"].mean()),
        "log_loss": float(rows["nll"].mean()),
        "recall_by_class": recalls.to_dict(),
        "rows": len(rows),
    }


def paired_prompt_bootstrap(
    predictions: pd.DataFrame, method: str, baseline: str, repeats: int, seed: int
) -> dict:
    selected = predictions[predictions["method"].isin([method, baseline])]
    rows = selected.groupby(["id", "prompt_id", "model", "method"])["correct"].mean()
    paired = rows.unstack("method")
    if paired.isna().any().any() or not {method, baseline}.issubset(paired.columns):
        raise ValueError("Paired comparisons require identical evaluated rows.")
    paired["difference"] = paired[method] - paired[baseline]
    paired = paired.reset_index()
    totals = pd.crosstab(paired["prompt_id"], paired["model"])
    gains = paired.pivot_table(
        index="prompt_id", columns="model", values="difference", aggfunc="sum", fill_value=0
    ).reindex(index=totals.index, columns=totals.columns)
    counts, differences = totals.to_numpy(), gains.to_numpy()
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repeats):
        indices = rng.integers(0, len(counts), size=len(counts))
        denominator = counts[indices].sum(axis=0)
        if (denominator == 0).any():
            continue
        samples.append(float((differences[indices].sum(axis=0) / denominator).mean()))
    if len(samples) < repeats * 0.95:
        raise ValueError("Too few valid bootstrap samples; some labels have too few prompts.")
    return {
        "method": method,
        "baseline": baseline,
        "delta_balanced_accuracy": float((differences.sum(axis=0) / counts.sum(axis=0)).mean()),
        "ci90": np.quantile(samples, [0.05, 0.95]).tolist(),
        "bootstrap_samples": len(samples),
        "prompts": len(counts),
        "scope": "Conditional on fitted models and these folds; not training-set uncertainty.",
    }


def summarize(predictions: pd.DataFrame, repeats: int, seed: int) -> dict:
    result = {
        "pooled": {name: metrics(rows) for name, rows in predictions.groupby("method")},
        "by_fold": {
            f"{fold}/{name}": metrics(rows)
            for (fold, name), rows in predictions.groupby(["fold", "method"])
        },
        "by_source": {
            f"{source}/{name}": metrics(rows)
            for (source, name), rows in predictions.groupby(["split", "method"])
        },
    }
    comparisons = []
    for size in (2, 3):
        method = f"shared_{size}"
        for baseline in (
            "independent",
            "logistic_full",
            "logistic_dense",
            "lda",
            f"centered_{size}",
            f"shuffled_{size}",
        ):
            comparisons.append(
                paired_prompt_bootstrap(predictions, method, baseline, repeats, seed)
            )
    result["comparisons"] = comparisons
    criteria = {}
    for size in (2, 3):
        method = f"shared_{size}"
        scores = result["pooled"]
        shared = scores[method]["balanced_accuracy"]
        gain = shared - scores["independent"]["balanced_accuracy"]
        comparison = next(
            item
            for item in comparisons
            if item["method"] == method and item["baseline"] == "independent"
        )
        criteria[method] = {
            "gain_at_least_one_point": gain >= 0.01,
            "positive_paired_ci90": comparison["ci90"][0] > 0,
            "beats_all_standalone_baselines": all(
                shared > scores[baseline]["balanced_accuracy"]
                for baseline in ("logistic_full", "logistic_dense", "lda")
            ),
            "beats_simple_centering": shared > scores[f"centered_{size}"]["balanced_accuracy"],
            "shuffle_removes_at_least_half_gain": (
                gain > 0 and shared - scores[f"shuffled_{size}"]["balanced_accuracy"] >= gain / 2
            ),
        }
    result["initial_gate_criteria"] = criteria
    return result
