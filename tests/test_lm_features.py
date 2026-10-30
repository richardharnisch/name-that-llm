"""Unit tests for masked pooling and layer parsing (no model download needed)."""

import numpy as np
import pytest
import torch

from name_that_llm.lm_features import _masked_max_abs, _masked_mean, parse_layers


def test_masked_mean_ignores_padding():
    states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    mask = torch.tensor([[1.0, 0.0]])
    np.testing.assert_allclose(_masked_mean(states, mask), [[1.0, 2.0]], rtol=1e-6)


def test_masked_mean_all_padding_clamps_denominator():
    states = torch.tensor([[[1.0, 1.0]]])
    mask = torch.zeros(1, 1)
    assert np.isfinite(_masked_mean(states, mask)).all()


def test_masked_max_abs_ignores_padding():
    states = torch.tensor([[[-3.0, 0.5], [8.0, 2.0]]])
    mask = torch.tensor([[1.0, 0.0]])
    np.testing.assert_allclose(_masked_max_abs(states, mask), [[3.0, 0.5]], rtol=1e-6)


def test_parse_layers_accepts_valid_and_rejects_conflicts():
    assert parse_layers("1,7,9", 9) == [1, 7, 9]
    with pytest.raises(ValueError, match="in"):
        parse_layers("1,10", 9)
    with pytest.raises(ValueError, match="duplicate"):
        parse_layers("3,3", 9)
    with pytest.raises(ValueError, match="empty"):
        parse_layers("", 9)
