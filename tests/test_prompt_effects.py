"""Check the probability calculation against an independent reference and controls."""

from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import block_diag
from scipy.special import softmax
from scipy.stats import multivariate_normal

from name_that_llm.research.evaluation import metrics, paired_prompt_bootstrap
from name_that_llm.research.features import load_embeddings
from name_that_llm.research.gaussian import SharedPromptGaussian
from name_that_llm.research.groups import fragment_prompts, shuffle_groups, validate_design


def known_model(prompt_cov=None):
    model = SharedPromptGaussian()
    model.classes_ = np.arange(2)
    model.means_ = np.array([[-1.0, 0.3], [1.2, -0.1]])
    model.noise_cov_ = np.array([[0.5, 0.1], [0.1, 0.3]])
    model.prompt_cov_ = np.array([[2.0, 0.4], [0.4, 1.0]]) if prompt_cov is None else prompt_cov
    model._prepare_inference()
    return model


@pytest.mark.parametrize("size", [1, 2, 3])
def test_exact_marginals_match_joint_gaussian(size):
    model = known_model()
    x = np.array([[0.5, 0.1], [1.2, 0.5], [-0.2, 0.3]])[:size]
    covariance = block_diag(*[model.noise_cov_] * size)
    covariance += np.kron(np.ones((size, size)), model.prompt_cov_)
    assignments = np.array(list(product(range(2), repeat=size)))
    joint = softmax(
        [
            multivariate_normal.logpdf(x.ravel(), model.means_[labels].ravel(), covariance)
            for labels in assignments
        ]
    )
    expected = np.array(
        [[joint[assignments[:, i] == k].sum() for k in range(2)] for i in range(size)]
    )
    assert_allclose(model.group_proba(x), expected, atol=1e-12)


def test_no_prompt_variance_reduces_to_independence():
    model = known_model(np.zeros((2, 2)))
    x = np.array([[0.5, 0.1], [1.2, 0.5], [-0.2, 0.3]])
    assert_allclose(model.group_proba(x), model.predict_proba(x), atol=1e-12)


def test_permutation_and_repeated_authors():
    model = known_model()
    x = np.array([[-1.0, 0.3], [-1.0, 0.3], [1.2, -0.1]])
    probabilities = model.group_proba(x)
    permutation = [2, 0, 1]
    assert_allclose(model.group_proba(x[permutation]), probabilities[permutation])
    assert_array_equal(probabilities.argmax(axis=1), [0, 0, 1])


def test_partition_must_cover_each_row_once():
    model = known_model()
    with pytest.raises(ValueError, match="partition"):
        model.predict_groups(np.zeros((3, 2)), [np.array([0, 1]), np.array([1, 2])])
    with pytest.raises(ValueError, match="group sizes"):
        model.group_proba(np.zeros((4, 2)))


def test_synthetic_shared_effect_is_learnable_and_shuffle_removes_gain():
    rng = np.random.default_rng(123)
    means = np.array([[-1.5, 0.0], [1.5, 0.0]])

    def sample(n):
        labels = rng.integers(0, 2, size=(n, 3))  # Repeated authors occur naturally.
        effects = rng.normal(size=(n, 1, 2)) * np.array([1.8, 0.7])
        x = means[labels] + effects + rng.normal(scale=0.35, size=(n, 3, 2))
        return x.reshape(-1, 2), labels.ravel(), np.repeat(np.arange(n), 3)

    x, y, prompts = sample(300)
    model = SharedPromptGaussian(shrinkage=0, max_iter=300).fit(x, y, prompts)
    xv, yv, pv = sample(600)
    groups = fragment_prompts(pv, 3, 42)
    shuffled = shuffle_groups(groups, np.zeros(len(xv)), 42)
    independent = np.mean(model.predict_proba(xv).argmax(axis=1) == yv)
    shared = np.mean(model.predict_groups(xv, groups).argmax(axis=1) == yv)
    random = np.mean(model.predict_groups(xv, shuffled).argmax(axis=1) == yv)
    assert shared > independent + 0.05
    assert shared > random + 0.05
    assert_allclose(model.means_, means, atol=0.25)
    assert_allclose(model.noise_cov_, np.eye(2) * 0.35**2, atol=0.04)


def test_fragmentation_and_source_matched_shuffle():
    prompts = np.repeat(np.arange(5), [3, 4, 5, 6, 7])
    sources = np.where(prompts < 2, "a", "b")
    groups = fragment_prompts(prompts, 3, 42)
    shuffled = shuffle_groups(groups, sources, 43)
    assert_array_equal(np.sort(np.concatenate(groups)), np.arange(len(prompts)))
    assert_array_equal(np.sort(np.concatenate(shuffled)), np.arange(len(prompts)))
    for original, random in zip(groups, shuffled, strict=True):
        assert len(set(prompts[original])) == 1
        assert_array_equal(sources[original], sources[random])


def test_leaking_prompt_is_rejected():
    frame = pd.DataFrame({"prompt_id": ["p", "p"], "fold": [0, 1]})
    with pytest.raises(ValueError, match="multiple folds"):
        validate_design(frame)


def test_embedding_ids_are_aligned_and_duplicates_rejected(tmp_path):
    path = tmp_path / "vectors.npz"
    np.savez(path, ids=["b", "a"], embeddings=[[2, 3], [0, 1]])
    assert_array_equal(load_embeddings(path, pd.Series(["a", "b"])), [[0, 1], [2, 3]])
    np.savez(path, ids=["a", "a"], embeddings=[[2, 3], [0, 1]])
    with pytest.raises(ValueError, match="exactly"):
        load_embeddings(path, pd.Series(["a", "b"]))


def test_bootstrap_resamples_prompts_and_does_not_count_repeats_as_new_rows():
    rows = []
    for prompt in range(8):
        for label in ("a", "b"):
            for method, correct in (("new", 1), ("old", 0)):
                for _ in range(3 if method == "new" else 1):
                    rows.append(
                        {
                            "id": f"{prompt}-{label}",
                            "prompt_id": str(prompt),
                            "model": label,
                            "method": method,
                            "correct": correct,
                            "nll": 0.1,
                        }
                    )
    predictions = pd.DataFrame(rows)
    result = paired_prompt_bootstrap(predictions, "new", "old", 100, 42)
    assert result["ci90"] == [1, 1]
    assert result["prompts"] == 8
    assert metrics(predictions[predictions["method"] == "new"])["rows"] == 16
