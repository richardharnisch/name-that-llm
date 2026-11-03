"""BPE tokenization with the interface used in the course tutorial."""

from collections.abc import Iterable

from tokenizers import Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer

DEFAULT_VOCAB_SIZE = 8000
DEFAULT_MIN_FREQUENCY = 2
UNK_TOKEN = "[UNK]"


def train_tokenizer(
    corpus: Iterable[str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
) -> Tokenizer:
    """Learn a BPE vocabulary and merge rules from training texts only."""
    if vocab_size < 2:
        raise ValueError("Vocabulary size must be at least 2.")
    if min_frequency < 1:
        raise ValueError("Minimum frequency must be positive.")
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Metaspace()
    tokenizer.decoder = MetaspaceDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[UNK_TOKEN],
        show_progress=False,
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    if tokenizer.get_vocab_size() < 2:
        raise ValueError("The training corpus contains no text to tokenize.")
    return tokenizer
