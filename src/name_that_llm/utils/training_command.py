"""Cross-validate PyTorch classifiers on BPE features or mean-pooled word embeddings."""

import argparse
from pathlib import Path
from time import perf_counter

from name_that_llm.classifiers import MODELS
from name_that_llm.config import (
    DATA_DIR,
    DEFAULT_MLP,
    DEFAULT_TOKENIZER,
    DEFAULT_TRAINING,
    TOKENIZER_ROOT,
    TRAIN_OUTPUT_DIR,
    MLPConfig,
    TokenizerConfig,
    TrainingConfig,
    save_config,
)
from name_that_llm.embeddings import PRETRAINED_MODELS, build_embedding_encoder
from name_that_llm.encodings import ENCODERS
from name_that_llm.experiment import run_fold
from name_that_llm.utils.data import load_folds
from name_that_llm.utils.training_report import save_cv_report


def run(args: argparse.Namespace) -> None:
    start = perf_counter()
    config = TrainingConfig(
        model=args.model,
        encoder=args.encoder,
        normalize=args.normalize,
        max_epochs=args.max_epochs,
        tol=args.tol,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        limit_per_class=args.limit_per_class,
    )
    mlp_config = MLPConfig(hidden=tuple(args.hidden), activation=DEFAULT_MLP.activation)
    tokenizer_config = TokenizerConfig(
        vocab_size=args.vocab_size, min_frequency=args.min_frequency
    )
    available = sorted(set(load_folds(args.data_dir)["fold"]))
    folds = available if args.fold is None else [args.fold]
    if not set(folds) <= set(available):
        raise ValueError(f"Fold {args.fold} does not exist. Folds: {available}.")

    word_encoder = None
    if config.encoder in PRETRAINED_MODELS:
        print(f"Loading {config.encoder} word vectors", flush=True)
        word_encoder = build_embedding_encoder(config.encoder)

    args.output.mkdir(parents=True, exist_ok=True)
    save_config(
        args.output / "config.json",
        **({"tokenizer": tokenizer_config} if word_encoder is None else {}),
        training=config,
        **({"mlp": mlp_config} if config.model == "mlp" else {}),
    )
    results, all_true, all_predicted = [], [], []
    for position, fold in enumerate(folds, start=1):
        print(f"\n===== Fold {fold} ({position}/{len(folds)}) =====", flush=True)
        metrics, true_names, predicted_names = run_fold(
            fold,
            config,
            mlp_config,
            tokenizer_config,
            data_dir=args.data_dir,
            tokenizer_root=args.tokenizer_root,
            tokenized_root=args.tokenized_root,
            output=args.output / f"fold-{fold}",
            word_encoder=word_encoder,
        )
        results.append(metrics)
        all_true += true_names
        all_predicted += predicted_names

    save_cv_report(args.output, folds, results, all_true, all_predicted, config, start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing folds.csv",
    )
    parser.add_argument(
        "--fold", type=int, help="Run one fold only (default: every fold)"
    )
    parser.add_argument("--tokenizer-root", type=Path, default=TOKENIZER_ROOT)
    parser.add_argument(
        "--tokenized-root",
        type=Path,
        help="Tokenized data root (default: <data-dir>/tokenized)",
    )
    parser.add_argument("--output", type=Path, default=TRAIN_OUTPUT_DIR)
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_TRAINING.model)
    parser.add_argument(
        "--encoder", choices=(*ENCODERS, *PRETRAINED_MODELS), default=DEFAULT_TRAINING.encoder
    )
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=list(DEFAULT_MLP.hidden),
        help="Hidden layer sizes, used only by --model mlp",
    )
    parser.add_argument(
        "--vocab-size",
        dest="vocab_size",
        type=int,
        default=DEFAULT_TOKENIZER.vocab_size,
        help="Requested BPE vocabulary size",
    )
    parser.add_argument(
        "--min-frequency", type=int, default=DEFAULT_TOKENIZER.min_frequency
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TRAINING.normalize,
        help="Scale each document vector to unit L2 norm",
    )
    parser.add_argument(
        "--max-epochs",
        "--epochs",
        dest="max_epochs",
        type=int,
        default=DEFAULT_TRAINING.max_epochs,
        help="Epoch cap. Training stops earlier when the training loss converges",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TRAINING.tol,
        help="Minimum relative training-loss improvement per epoch (0 = never stop early)",
    )
    parser.add_argument("--patience", type=int, default=DEFAULT_TRAINING.patience)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING.batch_size)
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_TRAINING.learning_rate
    )
    parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_TRAINING.weight_decay
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING.seed)
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=DEFAULT_TRAINING.limit_per_class,
        help="Cap rows per class in each set for a quick run",
    )
    args = parser.parse_args()
    if args.vocab_size < 2 or args.min_frequency < 1:
        parser.error(
            "--vocab-size must be at least 2 and --min-frequency must be positive."
        )
    if args.max_epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("--max-epochs, --batch-size and --patience must be positive.")
    if args.tol < 0:
        parser.error("--tol must not be negative.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error(
            "--learning-rate must be positive and --weight-decay must not be negative."
        )
    if any(size < 1 for size in args.hidden):
        parser.error("--hidden sizes must be positive.")
    if args.limit_per_class is not None and args.limit_per_class < 1:
        parser.error("--limit-per-class must be positive.")
    if args.fold is not None and args.fold < 0:
        parser.error("--fold must not be negative.")
    try:
        run(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Training failed: {error}\n")
