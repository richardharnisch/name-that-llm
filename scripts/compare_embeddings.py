"""Compare Gensim word embeddings inside one outer training split."""

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from gensim.models import KeyedVectors

from name_that_llm.classifiers import build_model
from name_that_llm.config import DATA_DIR, DEFAULT_MLP, DEFAULT_TRAINING
from name_that_llm.embeddings import (
    PRETRAINED_MODELS,
    WORD_TOKENIZER_NAME,
    build_embedding_encoder,
    encode_embedding,
    tokenize_embedding_words,
)
from name_that_llm.train import fit
from name_that_llm.utils.data import LABELS, assign_folds, load_folds, split_fold

PROBE_TOKENS = ("however", "therefore", "overall", "—", ";")
METRICS = ("validation_accuracy", "validation_macro_f1")


def coverage(counts: Counter[str], vectors: KeyedVectors) -> dict:
    missing = Counter({word: count for word, count in counts.items() if word not in vectors})
    return {
        "tokens": counts.total(),
        "types": len(counts),
        "token_coverage": 1 - missing.total() / counts.total(),
        "type_coverage": 1 - len(missing) / len(counts),
        "most_frequent_missing": missing.most_common(15),
    }


def run(args: argparse.Namespace) -> dict:
    train, held_out = split_fold(load_folds(args.data_dir), args.outer_fold)
    if set(train["prompt_id"]) & set(held_out["prompt_id"]):
        raise ValueError("Training and held-out data share a prompt ID.")
    development = assign_folds(train, n_folds=args.inner_folds, seed=DEFAULT_TRAINING.seed)
    texts = development["text"].tolist()
    folds = development["fold"].to_numpy()
    label_ids = {label: index for index, label in enumerate(LABELS)}
    labels = torch.tensor(development["model"].map(label_ids).to_numpy(), dtype=torch.long)
    counts = {
        "whitespace": Counter(word for text in texts for word in text.split()),
        "whitespace-lowercase": Counter(word for text in texts for word in text.lower().split()),
        WORD_TOKENIZER_NAME: Counter(
            word for text in texts for word in tokenize_embedding_words(text)
        ),
    }
    config = replace(DEFAULT_TRAINING, pooling="mean", max_epochs=args.epochs, tol=0)
    models = {}
    for name, model_id in PRETRAINED_MODELS.items():
        print(f"Loading {model_id}", flush=True)
        encoder = build_embedding_encoder(name)
        if len(encoder) != 300:
            raise ValueError(f"Expected 300 dimensions for {model_id}, found {len(encoder)}.")
        vocabulary = {
            method: coverage(tokens, encoder.vectors) for method, tokens in counts.items()
        }
        for method, stats in vocabulary.items():
            print(
                f"{name} {method}: token coverage {stats['token_coverage']:.2%}, "
                f"type coverage {stats['type_coverage']:.2%}",
                flush=True,
            )
        neighbors = {
            token: encoder.vectors.most_similar(token, topn=5) if token in encoder.vectors else None
            for token in PROBE_TOKENS
        }
        features = encode_embedding(texts, encoder, normalize=config.normalize)
        del encoder  # Release the full embedding table before loading the next one.
        run_config = replace(config, encoder=name)
        results = []
        for fold in range(args.inner_folds):
            val = folds == fold
            print(f"{name}: inner fold {fold} ({val.sum():,} validation passages)", flush=True)
            torch.manual_seed(config.seed)
            classifier = build_model(run_config, DEFAULT_MLP, features.shape[1], len(LABELS))
            history, _ = fit(
                classifier, features[~val], labels[~val], features[val], labels[val], run_config
            )
            results.append({"fold": fold, **{metric: history[-1][metric] for metric in METRICS}})
        models[name] = {
            "model_id": model_id,
            "coverage": vocabulary,
            "cosine_neighbors": neighbors,
            "scores_by_fold": results,
            "mean_scores": {
                metric: np.mean([row[metric] for row in results]) for metric in METRICS
            },
        }

    return {
        "held_out_outer_fold": args.outer_fold,
        "development_rows": len(development),
        "development_prompts": development["prompt_id"].nunique(),
        "validation_rows_per_inner_fold": np.bincount(folds).tolist(),
        "split_seed": DEFAULT_TRAINING.seed,
        "training": {**asdict(config), "encoder": "see models"},
        "tokenizer": WORD_TOKENIZER_NAME,
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--outer-fold", type=int, default=0, help="Hold this fold out of development"
    )
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("artifacts/embeddings/comparison.json"))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.set_num_threads(1)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
