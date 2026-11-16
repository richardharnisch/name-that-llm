# Prompt-effect experiment

Question: do other passages from the same unseen prompt improve author prediction
after we control for features, fitted parameters, and regularization?

This is an initial local experiment, not a competition submission or a novelty claim.
The existing assignment pipeline is unchanged.

## Model

For passage i from prompt g with author k, use

`x_i = mu_k + u_g + e_i`,

with a Gaussian prompt effect `u_g ~ N(0, P)` and independent noise `e_i ~ N(0, E)`.
The same prompt contributes covariance P between its passages. Training estimates
class means and both full covariance matrices with regularized EM. Each covariance
update shrinks 10% toward its own isotropic variance. This is a practical estimator;
the shrinkage step does not guarantee monotone maximum likelihood. Report convergence.

At inference, integrate out the prompt effect and marginalize companion labels.
For groups of up to three, enumerate all 18^n label assignments exactly. This permits
repeated authors and imposes no class-count constraints. Uniform author priors align
the decision rule with balanced accuracy under the assumed independent label prior.
For single passages, the covariance is P + E. The independent control uses exactly
the same trained means and covariances as the grouped method.

The assumptions are additive shared effects, transferable covariance, and a useful
Gaussian approximation in the feature space. Covariance caused by source, batch,
or author-by-prompt interactions can violate this interpretation.

## Fixed initial protocol

Use the existing version-7 public dataset and its original five prompt-disjoint
folds. Do not use reconstructed competition prompts. The runner rejects prompt
overlap, duplicate normalized passage text, and missing labels. This does not detect
semantic paraphrases or incorrect upstream prompt IDs.

1. Fit word TF-IDF (1–2 grams) or character TF-IDF (3–5 grams) with at most 20,000
   features on each training fold. Reduce to 64 dimensions with training-only SVD.
   Run both fixed feature views; do not choose settings using the outer fold scores.
2. Fit the shared Gaussian and its identical-parameter independent control.
3. Fit independent logistic regression on full and reduced features, and automatic
   shrinkage LDA on reduced features. Select logistic C from 0.1, 1, 10 using a
   training-only 20% prompt holdout. The unsupervised representation uses all outer
   training rows, including this inner holdout; outer validation never enters it.
4. Fragment each validation prompt into pairs and triples without using author
   labels. Retain remainders. Use seeds 42, 43, 44; average their *scores*, not their
   probabilities. Repetitions are not independent samples.
5. Shuffle companion membership within source, preserving group sizes. This controls
   for additional inference compute and source composition. Accidental same-prompt
   pairs are counted. Compare a naive baseline that subtracts observed group means
   and fits LDA on identically fragmented training groups. Its singleton fallback is
   ordinary LDA.
6. Report balanced accuracy, log loss, class recalls, source and fold results.
   Use 2,000 paired bootstrap draws of whole prompts for 90% difference intervals.
   These intervals condition on fitted models; they do not include training-set
   uncertainty or correct for trying multiple hypotheses. Fold training sets overlap.

Preliminary advance criterion, fixed before the full runs: for pairs and triples in
both feature views, gain at least 1 balanced-accuracy point over the matched
independent head, have a positive 90% paired interval, and beat each independent
baseline. Shuffling should remove at least half the gain over the matched head.
If simple centering matches or wins, the evidence does not support the extra
probabilistic machinery. Report failed criteria; do not tune until they pass.

This is a diagnostic gate for investing in frozen embeddings, not the original
publishability gate. A paper still needs strong frozen encoders, unknown prompt
groups, genuine repeated-author groups, source-shift evaluation, and a reproduction
of the prior shared-Gaussian competition method discussed in the research review.
Two lexical feature views do not count as two pretrained encoders.

## Run

```sh
uv sync --group dev
uv run python scripts/download_data.py
uv run python -m name_that_llm.research.run --features word --fold 0 \
  --partition-seeds 42 --output artifacts/prompt-effects/word-smoke
uv run python -m name_that_llm.research.run --features word \
  --output artifacts/prompt-effects/word
uv run python -m name_that_llm.research.run --features char \
  --output artifacts/prompt-effects/char
uv run pytest -q
uv run ruff check src/name_that_llm/research tests
```

The output directory must be new. Each run saves configuration, dependency versions,
input and source hashes, Git revision/status, per-fold passage predictions and group
IDs, metrics, and convergence diagnostics. Data and artifacts remain ignored by Git.
An interrupted run retains `status: running` and any completed fold predictions.
All results are development estimates on public data, not Kaggle test scores.

For later frozen embeddings, supply an NPZ file with `ids` (Unicode strings) and
`embeddings` (a finite 2D array), one per row of folds.csv. IDs are aligned explicitly.
PCA is fitted on training rows only. The full-vector logistic control is retained.

```sh
uv run python -m name_that_llm.research.run --features embeddings \
  --embeddings artifacts/frozen-vectors.npz \
  --embedding-provenance 'Exact checkpoint/revision, pooling, and training history' \
  --output artifacts/prompt-effects/frozen
```

Do not use an encoder or adapter trained on evaluation author labels. The provenance
argument documents this requirement; the code cannot establish an encoder's history.
No encoder weights are downloaded by the research runner.

## Code map

- `gaussian.py`: fit and exact probability calculation.
- `groups.py`: prompt fragmentation, shuffle control, and split checks.
- `features.py`: training-only text transforms and external vector alignment.
- `evaluation.py`: paired comparisons and score aggregation.
- `run.py`: experiment configuration, baselines, and saved evidence.
- `tests/test_prompt_effects.py`: joint-Gaussian reference, synthetic recovery,
  repeated authors, independence, permutation, grouping, and bootstrap checks.
