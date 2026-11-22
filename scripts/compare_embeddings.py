"""Compare Gensim word embeddings on the project's existing folds."""

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import torch
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastTextKeyedVectors

from name_that_llm.config import (
    DATA_DIR,
    DEFAULT_MLP,
    DEFAULT_TOKENIZER,
    DEFAULT_TRAINING,
    TOKENIZER_ROOT,
)
from name_that_llm.embeddings import (
    PRETRAINED_MODELS,
    WORD_TOKENIZER_NAME,
    build_embedding_encoder,
    tokenize_embedding_words,
)
from name_that_llm.experiment import run_fold
from name_that_llm.utils.data import load_folds
from name_that_llm.utils.training_report import summarise

PROBE_TOKENS = ("however", "therefore", "overall", "—", ";")
METRICS = ("validation_accuracy", "validation_macro_f1", "validation_log_loss")


def coverage(counts: Counter[str], vectors: KeyedVectors) -> dict:
    outside = Counter(
        {word: count for word, count in counts.items() if not vectors.has_index_for(word)}
    )
    missing = outside
    if isinstance(vectors, FastTextKeyedVectors):
        missing = Counter(
            {word: count for word, count in outside.items() if not vectors[word].any()}
        )
    return {
        "tokens": counts.total(),
        "types": len(counts),
        "token_coverage": 1 - missing.total() / counts.total(),
        "type_coverage": 1 - len(missing) / len(counts),
        "vocabulary_token_coverage": 1 - outside.total() / counts.total(),
        "vocabulary_type_coverage": 1 - len(outside) / len(counts),
        "composed_tokens": outside.total() - missing.total(),
        "composed_types": len(outside) - len(missing),
        "most_frequent_missing": missing.most_common(15),
        "most_frequent_outside_vocabulary": outside.most_common(15),
    }


def run(args: argparse.Namespace) -> dict:
    passages = load_folds(args.data_dir)
    fold_sizes = passages.groupby("fold").size().sort_index()
    folds = fold_sizes.index.tolist()
    texts = passages["text"].tolist()
    counts = {
        "whitespace": Counter(word for text in texts for word in text.split()),
        "whitespace-lowercase": Counter(word for text in texts for word in text.lower().split()),
        "nltk-word-lowercase": Counter(
            word for text in texts for word in tokenize_embedding_words(text, preserve_line=True)
        ),
        WORD_TOKENIZER_NAME: Counter(
            word for text in texts for word in tokenize_embedding_words(text)
        ),
    }
    config = replace(DEFAULT_TRAINING, pooling="mean", max_epochs=args.epochs)
    runs_dir = args.output.parent / f"{args.output.stem}-runs"
    models = {}
    for name, model_id in PRETRAINED_MODELS.items():
        if args.embedding and name not in args.embedding:
            continue
        print(f"Loading {model_id}", flush=True)
        encoder = build_embedding_encoder(name)
        if len(encoder) != 300:
            raise ValueError(f"Expected 300 dimensions for {model_id}, found {len(encoder)}.")
        vocabulary = {
            method: coverage(tokens, encoder.vectors) for method, tokens in counts.items()
        }
        for method, stats in vocabulary.items():
            print(
                f"{name} {method}: token coverage {stats['token_coverage']:.2%} "
                f"({stats['vocabulary_token_coverage']:.2%} direct), "
                f"type coverage {stats['type_coverage']:.2%} "
                f"({stats['vocabulary_type_coverage']:.2%} direct)",
                flush=True,
            )
        neighbors = {
            token: encoder.vectors.most_similar(token, topn=5) if token in encoder.vectors else None
            for token in PROBE_TOKENS
        }
        run_config = replace(config, encoder=name)
        results = []
        for fold in folds:
            print(f"{name}: fold {fold}", flush=True)
            metrics, _, _ = run_fold(
                fold,
                run_config,
                DEFAULT_MLP,
                DEFAULT_TOKENIZER,
                data_dir=args.data_dir,
                tokenizer_root=TOKENIZER_ROOT,
                tokenized_root=None,
                output=runs_dir / name / f"fold-{fold}",
                word_encoder=encoder,
            )
            results.append(metrics)
        del encoder  # Release the full embedding table before loading the next one.
        scores = summarise(results)
        models[name] = {
            "model_id": model_id,
            "coverage": vocabulary,
            "cosine_neighbors": neighbors,
            "scores_by_fold": [
                {key: row[key] for key in ("fold", "epochs_run", "converged", *METRICS)}
                for row in results
            ],
            "scores": scores,
        }

    return {
        "passages": len(passages),
        "prompts": passages["prompt_id"].nunique(),
        "folds": folds,
        "validation_rows_per_fold": fold_sizes.tolist(),
        "training": {**asdict(config), "encoder": "see models"},
        "tokenizer": WORD_TOKENIZER_NAME,
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING.max_epochs)
    parser.add_argument(
        "--embedding", choices=PRETRAINED_MODELS, action="append", help="Default: both embeddings."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/embeddings/comparison-binary.json")
    )
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
