"""Relate passage properties to the correctness of several classifiers.

Each system contributes one out-of-fold prediction per passage: the model of fold k
predicts the passages of fold k, which it did not train on. A Bayesian logistic
regression (the model of `nlp1-student/part2/analysis.py`) then estimates, for each
system, how each standardized passage feature shifts the probability of a correct
prediction. Laplace(0, 1) priors pull weights to 0 unless the data supports them.
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import textstat  # noqa: E402
from nltk.tokenize import sent_tokenize  # noqa: E402

from name_that_llm.config import DATA_DIR  # noqa: E402
from name_that_llm.embeddings import tokenize_embedding_words  # noqa: E402
from name_that_llm.utils.data import LABELS, load_folds  # noqa: E402

FEATURES = {
    "log_words": "log of the number of words",
    "sentence_length": "mean words per sentence",
    "lexical_diversity": "moving-average type-token ratio, 40-word window",
    "unseen_word_rate": "share of word tokens not in the training folds",
    "reading_ease": "Flesch reading ease",
    "em_dash_rate": "em dashes per 100 words",
    "semicolon_colon_rate": "semicolons and colons per 100 words",
    "curly_quotes": "contains a curly apostrophe or quotation mark",
    "straight_apostrophe": "contains a straight apostrophe",
}
WORD = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")


def moving_type_token_ratio(words: list[str], window: int = 40) -> float:
    """Mean type-token ratio of all windows, so long passages are not penalized."""
    if len(words) <= window:
        return len(set(words)) / max(1, len(words))
    counts = Counter(words[:window])
    ratios = [len(counts) / window]
    for start in range(1, len(words) - window + 1):
        old, new = words[start - 1], words[start + window - 1]
        counts[old] -= 1
        if counts[old] == 0:
            del counts[old]
        counts[new] += 1
        ratios.append(len(counts) / window)
    return float(np.mean(ratios))


def passage_features(passages: pd.DataFrame) -> pd.DataFrame:
    """Return the unstandardized features of each passage, indexed by passage ID.

    The unseen-word rate uses the training folds of the passage's own fold, which is
    the vocabulary its classifier saw.
    """
    tokens = [tokenize_embedding_words(text) for text in passages["text"]]
    unseen = np.zeros(len(passages))
    folds = passages["fold"].to_numpy()
    for fold in np.unique(folds):
        known = {word for document, f in zip(tokens, folds, strict=True) if f != fold
                 for word in document}
        for row in np.flatnonzero(folds == fold):
            unseen[row] = np.mean([word not in known for word in tokens[row]])

    rows = []
    for text in passages["text"]:
        words = WORD.findall(text.lower())
        n_words = max(1, len(text.split()))
        rows.append(
            {
                "log_words": np.log(n_words),
                "sentence_length": n_words / max(1, len(sent_tokenize(text))),
                "lexical_diversity": moving_type_token_ratio(words),
                "reading_ease": textstat.flesch_reading_ease(text),
                "em_dash_rate": 100 * text.count("—") / n_words,
                "semicolon_colon_rate": 100 * (text.count(";") + text.count(":")) / n_words,
                "curly_quotes": float(bool(re.search("[’‘“”]", text))),
                "straight_apostrophe": float("'" in text),
            }
        )
    frame = pd.DataFrame(rows, index=passages["id"])
    frame.insert(3, "unseen_word_rate", unseen)
    return frame


def bpe_predictions(run_dir: Path) -> pd.DataFrame:
    """Predict each fold's validation passages with the saved Part 1 fold models."""
    import torch

    from name_that_llm.encodings import encode_bpe, encoder_from_checkpoint
    from name_that_llm.train import predict_logits
    from name_that_llm.utils.checkpoints import load_checkpoint
    from name_that_llm.utils.tokenizer import Tokenizer

    parts = []
    for fold_dir in sorted(run_dir.glob("fold-*")):
        metrics = json.loads((fold_dir / "metrics.json").read_text())
        model, checkpoint = load_checkpoint(fold_dir / "model.pt")
        tokenizer = Tokenizer.from_file(checkpoint["tokenizer_path"])
        encoder = encoder_from_checkpoint(checkpoint, tokenizer)
        val = pd.read_parquet(Path(metrics["sample_directory"]) / "val.parquet")
        logits = predict_logits(model, encode_bpe(val["input_ids"], encoder), 256)
        log_probs = torch.log_softmax(logits, dim=1).numpy()
        parts.append(
            pd.DataFrame(
                {
                    "id": val["id"].to_numpy(),
                    "fold": metrics["fold"],
                    "label": val["model"].to_numpy(),
                    "predicted": [LABELS[i] for i in log_probs.argmax(axis=1)],
                    "confidence": np.exp(log_probs.max(axis=1)),
                }
            )
        )
    if not parts:
        raise ValueError(f"No fold-K directories in {run_dir}.")
    return pd.concat(parts, ignore_index=True)


def load_predictions(source: Path) -> pd.DataFrame:
    """Read saved sweep predictions, or predict with a directory of Part 1 fold models."""
    frame = pd.read_parquet(source) if source.is_file() else bpe_predictions(source)
    return frame[["id", "fold", "label", "predicted", "confidence"]]


def correctness_table(systems: dict[str, Path]) -> pd.DataFrame:
    """One row per passage and system, with a 0/1 correctness flag."""
    frames = []
    for name, source in systems.items():
        frame = load_predictions(source).assign(system=name)
        if not frame["id"].is_unique:
            raise ValueError(f"{name} predicts some passages more than once.")
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table["correct"] = (table["label"] == table["predicted"]).astype(int)
    counts = table.groupby("system")["id"].nunique()
    if counts.nunique() != 1:
        raise ValueError(f"Systems predict different passage counts: {counts.to_dict()}")
    return table


def fit_bda(
    table: pd.DataFrame,
    features: pd.DataFrame,
    *,
    samples: int = 1000,
    chains: int = 4,
    seed: int = 0,
):
    """Fit the course's correctness model with NUTS and return the posterior samples.

    mu_n = phi + delta[system_n] + w[system_n] . f_n, correct_n ~ Bernoulli(sigmoid(mu_n))
    """
    os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={chains}")
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    systems = list(dict.fromkeys(table["system"]))
    system_ids = jnp.asarray(table["system"].map({s: i for i, s in enumerate(systems)}))
    x = jnp.asarray(features.loc[table["id"]].to_numpy(), dtype=jnp.float32)
    y = jnp.asarray(table["correct"].to_numpy())

    def model(system_ids, x, y=None):
        phi = numpyro.sample("phi", dist.Laplace(0.0, 1.0))
        with numpyro.plate("systems", len(systems)):
            delta = numpyro.sample("delta", dist.Laplace(0.0, 1.0))
            w = numpyro.sample(
                "w", dist.Laplace(0.0, 1.0).expand([x.shape[1]]).to_event(1)
            )
        mu = phi + delta[system_ids] + jnp.sum(w[system_ids] * x, axis=-1)
        numpyro.deterministic("p", jax.nn.sigmoid(mu))
        with numpyro.plate("data", x.shape[0]):
            numpyro.sample("correct", dist.Bernoulli(logits=mu), obs=y)

    mcmc = MCMC(
        NUTS(model),
        num_warmup=samples,
        num_samples=samples,
        num_chains=chains,
        chain_method="parallel",
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), system_ids, x, y)
    return mcmc, systems


def rope_probabilities(values: np.ndarray, rope: float) -> dict[str, float]:
    """Posterior mass below, inside and above the region of practical equivalence."""
    return {
        "p_below_rope": float((values < -rope).mean()),
        "p_in_rope": float((np.abs(values) <= rope).mean()),
        "p_above_rope": float((values > rope).mean()),
    }


def summarise_posterior(
    mcmc, systems, feature_names, table, rope: float
) -> tuple[pd.DataFrame, dict]:
    """Posterior summaries of the feature weights, system accuracy and diagnostics.

    Each weight and each difference of weights gets the posterior mass below, inside and
    above [-rope, rope].
    """
    from numpyro.diagnostics import summary

    draws = mcmc.get_samples(group_by_chain=True)
    stats = summary({k: v for k, v in draws.items() if k != "p"})
    flat = mcmc.get_samples()
    w = np.asarray(flat["w"])  # (draws, systems, features)
    rows = []
    for s, system in enumerate(systems):
        for f, feature in enumerate(feature_names):
            values = w[:, s, f]
            low, high = np.quantile(values, [0.05, 0.95])
            rows.append(
                {
                    "system": system,
                    "feature": feature,
                    "mean": values.mean(),
                    "q05": low,
                    "q95": high,
                    "p_positive": (values > 0).mean(),
                    **rope_probabilities(values, rope),
                    "r_hat": float(stats["w"]["r_hat"][s, f]),
                }
            )
    for a, first in enumerate(systems):
        for b, second in enumerate(systems[a + 1 :], start=a + 1):
            for f, feature in enumerate(feature_names):
                difference = w[:, a, f] - w[:, b, f]
                rows.append(
                    {
                        "system": f"{first} - {second}",
                        "feature": feature,
                        "mean": difference.mean(),
                        "q05": np.quantile(difference, 0.05),
                        "q95": np.quantile(difference, 0.95),
                        "p_positive": (difference > 0).mean(),
                        **rope_probabilities(difference, rope),
                        "r_hat": np.nan,
                    }
                )
    p = np.asarray(flat["p"])
    accuracy = {}
    for system in systems:
        rows_of_system = (table["system"] == system).to_numpy()
        values = p[:, rows_of_system].mean(axis=1)
        accuracy[system] = {
            "observed": float(table.loc[rows_of_system, "correct"].mean()),
            "posterior_mean": float(values.mean()),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
        }
    divergences = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    max_r_hat = max(float(np.max(v["r_hat"])) for v in stats.values())
    return pd.DataFrame(rows), {
        "accuracy": accuracy,
        "divergences": divergences,
        "max_r_hat": max_r_hat,
    }


def plot_weights(weights: pd.DataFrame, path: Path, rope: float | None = None) -> None:
    """Forest plot: one row per feature, one marker per system, 90% intervals."""
    features = list(dict.fromkeys(weights["feature"]))
    systems = list(dict.fromkeys(weights["system"]))
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(features) + 1.2))
    offsets = np.linspace(-0.25, 0.25, len(systems))
    for offset, system in zip(offsets, systems, strict=True):
        rows = weights[weights["system"] == system].set_index("feature").loc[features]
        y = np.arange(len(features)) + offset
        ax.errorbar(
            rows["mean"],
            y,
            xerr=[rows["mean"] - rows["q05"], rows["q95"] - rows["mean"]],
            fmt="o",
            markersize=4,
            capsize=2,
            label=system,
        )
    ax.axvline(0, color="grey", linewidth=0.8)
    if rope:
        ax.axvspan(-rope, rope, color="grey", alpha=0.15, label=f"ROPE ±{rope:g}")
    ax.set_yticks(np.arange(len(features)), features)
    ax.invert_yaxis()
    ax.set_xlabel("weight on logit of P(correct), per standard deviation (90% interval)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def binned_accuracy(table: pd.DataFrame, raw: pd.DataFrame, bins: int = 4) -> pd.DataFrame:
    """Observed accuracy per system in quartiles (or 0/1 groups) of each raw feature."""
    rows = []
    quartiles = [f"Q{i + 1}" for i in range(bins)]
    for feature in raw.columns:
        values = raw[feature]
        groups = (
            values.astype(int).astype(str)
            if values.nunique() <= 2
            else pd.qcut(values.rank(method="first"), bins, labels=quartiles)
        )
        merged = table.assign(group=groups.loc[table["id"]].to_numpy())
        result = merged.groupby(["group", "system"], observed=True)["correct"].agg(["mean", "size"])
        for (group, system), row in result.iterrows():
            rows.append(
                {
                    "feature": feature,
                    "group": str(group),
                    "system": system,
                    "accuracy": row["mean"],
                    "passages": int(row["size"]),
                }
            )
    return pd.DataFrame(rows)


def plot_binned(binned: pd.DataFrame, systems: list[str], path: Path) -> None:
    features = list(dict.fromkeys(binned["feature"]))
    columns = 3
    rows = int(np.ceil(len(features) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(11, 2.8 * rows), sharey=True)
    for ax, feature in zip(axes.flat, features, strict=False):
        part = binned[binned["feature"] == feature]
        for system in systems:
            group = part[part["system"] == system]
            ax.plot(group["group"], group["accuracy"], marker="o", label=system)
        ax.set_title(feature, fontsize=9)
        ax.set_ylim(0, None)
    for ax in axes.flat[len(features):]:
        ax.axis("off")
    axes.flat[0].set_ylabel("accuracy")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def agreement(table: pd.DataFrame) -> pd.DataFrame:
    """How often each pattern of correct (1) and wrong (0) systems occurs."""
    wide = table.pivot(index="id", columns="system", values="correct")
    wide = wide[list(dict.fromkeys(table["system"]))]
    pattern = wide.astype(str).agg("".join, axis=1)
    counts = pattern.value_counts().rename("passages").to_frame()
    counts.index.name = "correct pattern (" + ", ".join(wide.columns) + ")"
    counts["share"] = counts["passages"] / len(wide)
    return counts


def parse_system(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("Use NAME=PATH.")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--system",
        type=parse_system,
        action="append",
        required=True,
        help="NAME=PATH, where PATH is a predictions .parquet from embedding_sweep.py "
        "--save-predictions or a Part 1 run directory with fold-K/model.pt.",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=Path("artifacts/error-analysis/report"))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument(
        "--features",
        nargs="+",
        choices=list(FEATURES),
        default=list(FEATURES),
        help="Passage features to use (default: all).",
    )
    parser.add_argument(
        "--control-label",
        action="store_true",
        help="Add indicators of the true author model, so text-feature weights compare "
        "passages of the same model. Without this, a feature can look useful only "
        "because it marks a model that is easy to identify.",
    )
    parser.add_argument(
        "--rope",
        type=float,
        default=0.05,
        help="Half-width of the region of practical equivalence for weights and weight "
        "differences, in log-odds per standard deviation. At an accuracy p, a weight w "
        "changes accuracy by about p(1 - p) w, so 0.05 is about 1 accuracy point at p = 0.28, "
        "the same ROPE as the Part 1 comparison.",
    )
    args = parser.parse_args()

    systems = dict(args.system)
    args.output.mkdir(parents=True, exist_ok=True)
    passages = load_folds(args.data_dir)
    table = correctness_table(systems)
    if set(table["id"]) != set(passages["id"]):
        raise ValueError("The systems must predict every passage in folds.csv exactly once.")
    raw = passage_features(passages)[args.features]
    standardized = (raw - raw.mean()) / raw.std()

    raw.to_csv(args.output / "features.csv")
    table.to_csv(args.output / "correctness.csv", index=False)
    binned = binned_accuracy(table, raw)
    binned.to_csv(args.output / "binned_accuracy.csv", index=False)
    plot_binned(binned, list(systems), args.output / "binned_accuracy.png")
    agreement(table).to_csv(args.output / "agreement.csv")
    per_label = table.pivot_table(index="label", columns="system", values="correct",
                                  sort=False).loc[list(LABELS)]
    per_label[list(systems)].to_csv(args.output / "per_label_accuracy.csv")
    raw.corr().to_csv(args.output / "feature_correlations.csv")

    covariates = standardized
    if args.control_label:
        # The first label is the reference level, absorbed by phi and delta.
        labels = passages.set_index("id")["model"]
        indicators = pd.DataFrame(
            {f"label={label}": (labels == label).astype(float) for label in LABELS[1:]}
        )
        covariates = standardized.join(indicators)
    mcmc, system_names = fit_bda(table, covariates, samples=args.samples, chains=args.chains)
    weights, summary = summarise_posterior(
        mcmc, system_names, list(covariates.columns), table, args.rope
    )
    pairs = weights["system"].isin(system_names)
    differences = weights[~pairs & weights["feature"].isin(raw.columns)].drop(columns="r_hat")
    weights = weights[pairs]
    weights.to_csv(args.output / "weights.csv", index=False)
    differences.to_csv(args.output / "weight_differences.csv", index=False)
    plot_weights(
        weights[weights["feature"].isin(raw.columns)], args.output / "weights.png", args.rope
    )
    summary["control_label"] = args.control_label
    summary["rope"] = args.rope
    summary["features"] = {name: FEATURES[name] for name in args.features}
    summary["systems"] = {name: str(path) for name, path in systems.items()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary["accuracy"], indent=2))
    print(f"Divergences: {summary['divergences']}; max r_hat: {summary['max_r_hat']:.3f}")
    print(weights[weights["feature"].isin(raw.columns)].round(3).to_string(index=False))
    print(differences.round(3).to_string(index=False))
    print(f"Report in {args.output}")


if __name__ == "__main__":
    main()
