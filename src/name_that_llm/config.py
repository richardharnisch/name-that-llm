"""Experiment defaults, separate from paths and fixed dataset labels."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_SEED = 42


@dataclass(frozen=True)
class SplitConfig:
    # Exclude reconstructed competition prompts, the pilot, and the Kimi-only rerun.
    splits: tuple[str, ...] = ("new", "new2", "proxy_v1")
    val_size: float = 0.2
    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int = 8000
    min_frequency: int = 2


@dataclass(frozen=True)
class TrainingConfig:
    model: str = "log-linear"  # "log-linear", "mlp" or "svm"
    encoder: str = "bag-of-words"
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = DEFAULT_SEED
    limit_per_class: int | None = None


@dataclass(frozen=True)
class MLPConfig:
    hidden: tuple[int, ...] = (256,)
    activation: str = "ReLU"


DEFAULT_SPLIT = SplitConfig()
DEFAULT_TOKENIZER = TokenizerConfig()
DEFAULT_TRAINING = TrainingConfig()
DEFAULT_MLP = MLPConfig()

# Paths are not tunable hyperparameters.
DATA_DIR = Path("data")
TOKENIZER_ROOT = Path("artifacts/tokenizers")
TRAIN_OUTPUT_DIR = Path("artifacts/simple_train")


def save_config(
    path: Path, **groups: SplitConfig | TokenizerConfig | TrainingConfig | MLPConfig
) -> None:
    """Save the settings actually used, without rewriting an unchanged snapshot."""
    content = json.dumps({name: asdict(config) for name, config in groups.items()}, indent=2) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
