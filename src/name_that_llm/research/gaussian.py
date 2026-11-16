"""A shared Gaussian prompt effect, fitted on training prompts only.

For passage i in prompt g: x_i = mean[label_i] + u_g + e_i,
where u_g ~ N(0, prompt_cov) and e_i ~ N(0, noise_cov).
Inference integrates u_g and sums over ALL companion labels, including repeats.
"""

from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.special import logsumexp, softmax

Array = NDArray[np.float64]


def _regularize(covariance: Array, shrinkage: float, floor: float) -> Array:
    dimension = len(covariance)
    covariance = (covariance + covariance.T) / 2
    target = np.trace(covariance) / dimension * np.eye(dimension)
    values, vectors = eigh((1 - shrinkage) * covariance + shrinkage * target)
    return (vectors * np.maximum(values, floor)) @ vectors.T


class SharedPromptGaussian:
    """Full covariances with fixed isotropic shrinkage and uniform class priors.

    Regularized EM estimates means and covariances. Shrinkage after each covariance
    update is a practical estimator, not an exact maximum-likelihood EM update.
    The independent control uses these SAME parameters and integrates one prompt
    effect per passage. It adds no new parameters or tuning.
    """

    def __init__(self, shrinkage: float = 0.1, max_iter: int = 500, tol: float = 1e-5):
        if not 0 <= shrinkage <= 1 or max_iter < 1 or tol <= 0:
            raise ValueError("Invalid shrinkage, iteration limit, or tolerance.")
        self.shrinkage = shrinkage
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, x: Array, y: NDArray, prompts: NDArray) -> "SharedPromptGaussian":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or len(x) != len(y) or len(x) != len(prompts):
            raise ValueError("Features, labels, and prompts must have matching rows.")
        if not np.isfinite(x).all():
            raise ValueError("Features must be finite.")
        self.classes_, labels = np.unique(y, return_inverse=True)
        _, groups = np.unique(prompts, return_inverse=True)
        sizes = np.bincount(groups)
        if len(sizes) < 2 or len(sizes) == len(x):
            raise ValueError("Need multiple prompts and within-prompt replication.")
        n, dimension = x.shape
        means = np.array([x[labels == k].mean(axis=0) for k in range(len(self.classes_))])
        residual = x - means[labels]
        total = residual.T @ residual / n
        floor = max(float(np.trace(total) / dimension) * 1e-6, 1e-12)
        noise = _regularize(total / 2, self.shrinkage, floor)
        prompt = noise.copy()
        self.converged_ = False
        for iteration in range(1, self.max_iter + 1):
            self.n_iter_ = iteration
            noise_inv = np.linalg.inv(noise)
            prompt_inv = np.linalg.inv(prompt)
            # Posterior covariance depends on group size, not its text or label.
            unique_sizes, size_index = np.unique(sizes, return_inverse=True)
            posterior_cov = np.array(
                [np.linalg.inv(prompt_inv + size * noise_inv) for size in unique_sizes]
            )[size_index]
            sums = np.zeros((len(sizes), dimension))
            np.add.at(sums, groups, x - means[labels])
            effects = np.einsum("gij,gj->gi", posterior_cov, sums @ noise_inv)
            adjusted = x - effects[groups]
            new_means = np.array([adjusted[labels == k].mean(axis=0) for k in range(len(means))])
            residual = adjusted - new_means[labels]
            new_noise = _regularize(
                (residual.T @ residual + np.einsum("g,gij->ij", sizes, posterior_cov)) / n,
                self.shrinkage,
                floor,
            )
            new_prompt = _regularize(
                (effects.T @ effects + posterior_cov.sum(axis=0)) / len(sizes),
                self.shrinkage,
                floor,
            )
            changes = [
                np.linalg.norm(new - old) / max(np.linalg.norm(old), floor)
                for new, old in ((new_means, means), (new_noise, noise), (new_prompt, prompt))
            ]
            means, noise, prompt = new_means, new_noise, new_prompt
            self.relative_change_ = float(max(changes))
            if self.relative_change_ < self.tol:
                self.converged_ = True
                break
        self.means_ = means
        self.noise_cov_ = noise
        self.prompt_cov_ = prompt
        self._prepare_inference()
        return self

    def _prepare_inference(self) -> None:
        # Columns of transform whiten noise and diagonalize prompt covariance:
        # T.T @ noise @ T = I, T.T @ prompt @ T = diag(prompt_variance).
        values, self.transform_ = eigh(self.prompt_cov_, self.noise_cov_)
        self.prompt_variance_ = np.maximum(values, 0)
        self.transformed_means_ = self.means_ @ self.transform_

    def predict_proba(self, x: Array) -> Array:
        residual = np.asarray(x) @ self.transform_
        residual = residual[:, None, :] - self.transformed_means_[None, :, :]
        scores = -0.5 * np.sum(residual**2 / (1 + self.prompt_variance_), axis=2)
        return softmax(scores, axis=1)

    def group_proba(self, x: Array) -> Array:
        """Exact label marginals for one group of at most three passages.

        After whitening, the precision of a size-n group in dimension d is
        I - lambda_d / (1 + n * lambda_d) * 11'. The determinant is independent
        of label assignments, so it cancels in the posterior normalization.
        """
        size = len(x)
        if not 1 <= size <= 3:
            raise ValueError("Exact inference supports group sizes 1, 2, and 3 only.")
        if size == 1:
            return self.predict_proba(x)
        classes = len(self.classes_)
        residual = np.asarray(x) @ self.transform_
        residual = residual[:, None, :] - self.transformed_means_[None, :, :]
        # K**3 assignments, no constraints on repeated author labels.
        assignments = np.array(list(product(range(classes), repeat=size)))
        sums = np.zeros((len(assignments), residual.shape[-1]))
        squares = np.zeros(len(assignments))
        for position in range(size):
            selected = residual[position, assignments[:, position]]
            sums += selected
            squares += np.sum(selected**2, axis=1)
        correction = self.prompt_variance_ / (1 + size * self.prompt_variance_)
        scores = -0.5 * (squares - np.sum(sums**2 * correction, axis=1))
        normalizer = logsumexp(scores)
        return np.array(
            [
                [
                    np.exp(logsumexp(scores[assignments[:, i] == k]) - normalizer)
                    for k in range(classes)
                ]
                for i in range(size)
            ]
        )

    def predict_groups(self, x: Array, groups: list[NDArray]) -> Array:
        covered = np.concatenate(groups) if groups else np.array([], dtype=int)
        if not np.array_equal(np.sort(covered), np.arange(len(x))):
            raise ValueError("Groups must partition all input rows exactly once.")
        probabilities = np.empty((len(x), len(self.classes_)))
        for indices in groups:
            probabilities[indices] = self.group_proba(x[indices])
        return probabilities
