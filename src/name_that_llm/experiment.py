"""Assemble and run the experiment for one cross-validation fold."""

from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from name_that_llm.classifiers import build_model
from name_that_llm.config import MLPConfig, TokenizerConfig, TrainingConfig
from name_that_llm.embeddings import (
    WordEmbeddingEncoder,
    build_embedding_encoder,
    encode_embedding,
)
from name_that_llm.encodings import ENCODERS, build_encoder, encode_bpe
from name_that_llm.preprocess import load_fold_rows, preprocess_data
from name_that_llm.train import fit, predict_logits
from name_that_llm.utils.checkpoints import save_checkpoint
from name_that_llm.utils.data import LABELS
from name_that_llm.utils.tokenizer import Tokenizer
from name_that_llm.utils.training_report import (
    save_fold_metrics,
    save_learning_curves,
    score_fold,
)


def run_fold(
    fold: int,
    config: TrainingConfig,
    mlp_config: MLPConfig,
    tokenizer_config: TokenizerConfig,
    *,
    data_dir: Path,
    tokenizer_root: Path,
    tokenized_root: Path | None,
    output: Path,
    word_encoder: WordEmbeddingEncoder | None = None,
) -> tuple[dict, list[str], list[str]]:
    """Prepare features, train a classifier, and evaluate it on the held-out fold."""
    start = perf_counter()
    tokenizer_path = None
    if config.encoder in ENCODERS:
        if config.pooling is not None:
            raise ValueError("--pooling requires --encoder glove or fasttext.")
        tokenizer_path, sample_directory, reused = preprocess_data(
            directory=data_dir,
            tokenizer_root=tokenizer_root,
            output=tokenized_root,
            vocab_size=tokenizer_config.vocab_size,
            min_frequency=tokenizer_config.min_frequency,
            fold=fold,
        )
        print(
            f"{'Reused' if reused else 'Created'} BPE tokenizer and samples: {tokenizer_path}",
            flush=True,
        )
        train = pd.read_parquet(sample_directory / "train.parquet")
        val = pd.read_parquet(sample_directory / "val.parquet")
        feature_metadata = {
            "tokenizer": "bpe",
            "tokenizer_path": str(tokenizer_path),
            "sample_directory": str(sample_directory),
            "requested_vocab_size": tokenizer_config.vocab_size,
            "min_frequency": tokenizer_config.min_frequency,
        }
    else:
        frames = load_fold_rows(data_dir, fold)
        label_ids = {label: index for index, label in enumerate(LABELS)}
        train, val = (
            frame.assign(label=frame["model"].map(label_ids))
            for frame in (frames["train"], frames["val"])
        )
        feature_metadata = {"tokenizer": "whitespace"}
    if config.limit_per_class is not None:
        train, val = (
            frame.groupby("model", group_keys=False).sample(
                n=min(config.limit_per_class, frame.groupby("model").size().min()),
                random_state=config.seed,
            )
            for frame in (train, val)
        )

    print(f"Train: {len(train):,} rows; validation: {len(val):,} rows", flush=True)
    if tokenizer_path is not None:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        encoder = build_encoder(config.encoder, tokenizer, train["text"], config.normalize)
        print(f"Encoder: {config.encoder}; vocabulary: {len(encoder):,} tokens", flush=True)
        x_train = encode_bpe(train["input_ids"], encoder)
        x_val = encode_bpe(val["input_ids"], encoder)
        feature_metadata["vocab_size"] = len(encoder)
    else:
        encoder = word_encoder or build_embedding_encoder(config.encoder)
        pooling = config.pooling or "mean"
        print(
            f"Encoder: {config.encoder}; {pooling} pooling, {len(encoder)} dimensions", flush=True
        )
        x_train = encode_embedding(
            train["text"].tolist(), encoder, pooling=pooling, normalize=config.normalize
        )
        x_val = encode_embedding(
            val["text"].tolist(), encoder, pooling=pooling, normalize=config.normalize
        )
        feature_metadata["pooling"] = pooling
        feature_metadata["embedding_dim"] = len(encoder)
    y_train = torch.tensor(train["label"].to_numpy(), dtype=torch.long)
    y_val = torch.tensor(val["label"].to_numpy(), dtype=torch.long)

    torch.manual_seed(config.seed)
    model = build_model(config, mlp_config, len(encoder), len(LABELS))
    print(
        f"Training {config.model} ({sum(p.numel() for p in model.parameters()):,} parameters)"
    )
    history, converged = fit(model, x_train, y_train, x_val, y_val, config)
    print(
        f"{'Converged' if converged else 'Did NOT converge'} after {len(history)} epochs "
        f"(cap {config.max_epochs})",
        flush=True,
    )

    train_logits = predict_logits(model, x_train, config.batch_size)
    val_logits = predict_logits(model, x_val, config.batch_size)
    scores, true_names, predicted_names = score_fold(
        train_logits, y_train, val_logits, y_val, config.model
    )
    metrics = save_fold_metrics(
        output,
        fold,
        config,
        mlp_config,
        metadata={
            **feature_metadata,
            "train_rows": len(train),
            "validation_rows": len(val),
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        scores=scores,
        history=history,
        converged=converged,
        elapsed_seconds=perf_counter() - start,
    )
    save_checkpoint(
        output / "model.pt",
        model,
        config,
        mlp_config,
        tokenizer_config,
        tokenizer_path,
        encoder,
    )
    save_learning_curves(history, output)
    return metrics, true_names, predicted_names
