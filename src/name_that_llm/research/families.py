"""Descriptive error decomposition; name-based groups are not model genealogy."""

import argparse
import json
from pathlib import Path

import pandas as pd


def provider(label):
    return {
        "claude": "Anthropic",
        "gpt": "OpenAI",
        "gemini": "Google",
        "glm": "Zhipu",
        "kimi": "Moonshot",
        "qwen3.8": "Alibaba",
    }[label.split("-")[0]]


def family(label):
    if label.startswith("claude-"):
        return "-".join(label.split("-")[:2])
    if label.startswith("gpt-5.6-"):
        return "gpt-5.6"
    if label.startswith("gemini-") and label.endswith("-flash"):
        return "gemini-flash"
    if label.startswith("glm-5.3"):
        return "glm-5.3"
    return label


def summarize(frame):
    frame = frame.copy()
    frame["exact"] = frame.model == frame.prediction
    frame["same_family"] = frame.model.map(family) == frame.prediction.map(family)
    frame["same_provider"] = frame.model.map(provider) == frame.prediction.map(provider)
    frame["sibling_error"] = ~frame.exact & frame.same_family
    frame["other_family_error"] = ~frame.same_family & frame.same_provider
    frame["other_provider_error"] = ~frame.same_provider
    columns = [
        "exact",
        "same_family",
        "same_provider",
        "sibling_error",
        "other_family_error",
        "other_provider_error",
    ]
    # Equal weight per exact author, not per provider or repeated partition.
    rows = frame.groupby(["id", "model"])[columns].mean().reset_index()
    per_author = rows.groupby("model")[columns].mean()
    return {
        "macro_over_authors": per_author.mean().to_dict(),
        "per_author": per_author.to_dict(orient="index"),
        "per_family_exact_recall": per_author.exact.groupby(per_author.index.map(family))
        .mean()
        .to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    frame = pd.concat(
        [pd.read_parquet(path) for path in sorted(args.run.glob("fold-*-predictions.parquet"))]
    )
    result = {method: summarize(rows) for method, rows in frame.groupby("method")}
    (args.run / "family_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v["macro_over_authors"] for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
