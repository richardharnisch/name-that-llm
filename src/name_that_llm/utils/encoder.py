"""Interface for encoders that accept raw text."""

from abc import ABC, abstractmethod

import numpy as np


class Encoder(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """Return the width of each feature vector."""

    @abstractmethod
    def __call__(self, x: str) -> np.ndarray:
        """Encode raw text as one vector or a sequence with width len(self)."""
