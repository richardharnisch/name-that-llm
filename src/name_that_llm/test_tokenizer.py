"""Temporary word-and-punctuation tokenizer for pipeline experiments."""

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class Encoding:
    ids: list[int]


class Tokenizer:
    """Fit a small vocabulary on training text; map unseen tokens to ID 0."""

    def __init__(self, max_vocab: int = 5000):
        if max_vocab < 2:
            raise ValueError("max_vocab must be at least 2, including the unknown token.")
        self.max_vocab = max_vocab
        self._vocab = {"[UNK]": 0}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def fit(self, texts: Iterable[str]) -> "Tokenizer":
        counts = Counter(token for text in texts for token in self.tokenize(text))
        if not counts:
            raise ValueError("Cannot fit a tokenizer without any tokens.")
        self._vocab = {"[UNK]": 0}
        for token, _ in counts.most_common(self.max_vocab - 1):
            self._vocab[token] = len(self._vocab)
        return self

    def encode(self, text: str) -> Encoding:
        return Encoding([self._vocab.get(token, 0) for token in self.tokenize(text)])

    def get_vocab(self) -> dict[str, int]:
        return self._vocab.copy()
