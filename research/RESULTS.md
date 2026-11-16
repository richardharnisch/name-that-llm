# Initial local result: the information helps, but the method fails the baseline gate

Run date: 2026-11-16. Implementation commit: `974aecc`.
The exact scores, comparisons, configuration, and hashes are in
[results/initial.json](results/initial.json). Full passage predictions and diagnostics
remain local under `artifacts/prompt-effects/{word,char}/`.

## Decision

**No-go for adding prompt clustering or claiming a competitive method.** Known prompt
groups help the shared Gaussian, but full-feature logistic regression wins by a wide
margin. All four size/feature combinations fail the predeclared requirement to beat
the independent baselines. No outer-validation settings were changed to rescue them.

The next scientific check, if this direction continues, is the same comparison using
frozen pretrained embeddings with documented provenance. The runner already accepts
those vectors. Improving group recovery before fixing the representation would not
address the failure observed here.

## Evidence

Five prompt-disjoint folds, 4,818 public passages, 282 prompts, 18 authors. Word and
character features are separate fixed runs. Each uses 64-dimensional training-only
SVD, except the full-feature logistic control. Grouped results average three random
partitions without ensembling predictions. Scores below are pooled balanced accuracy
in percent, not the unweighted mean of fold scores.

| Method | Word features | Character features |
|---|---:|---:|
| Logistic regression, full features | **26.44** | **27.80** |
| Logistic regression, 64 dimensions | 13.25 | 16.76 |
| Shrinkage LDA, 64 dimensions | 14.44 | 17.27 |
| Matched independent Gaussian | 14.46 | 17.21 |
| Shared Gaussian, true pairs | 16.73 | 19.78 |
| Shared Gaussian, true triples | 19.04 | 21.30 |
| Centering + LDA, true pairs | 14.97 | 16.94 |
| Centering + LDA, true triples | 17.92 | 19.60 |
| Shared Gaussian, shuffled pairs | 12.74 | 14.65 |
| Shared Gaussian, shuffled triples | 13.37 | 15.35 |

Paired improvements over the identical-parameter independent Gaussian, in percentage
points; brackets are 90% prompt-bootstrap intervals:

| Group | Word features | Character features |
|---|---:|---:|
| Pairs | +2.27 [1.47, 3.10] | +2.57 [1.82, 3.35] |
| Triples | +4.58 [3.65, 5.48] | +4.08 [3.24, 4.94] |

The triple gain is positive in all five folds for both representations. It also
beats simple centering: +1.12 points [0.61, 1.63] for words and +1.69 [1.14, 2.24]
for characters. This is evidence against centering alone explaining the observed
gain. It is not a comparison with every possible centering-based classifier.

Against full-feature logistic regression, triples lose by 7.40 points
[-8.38, -6.38] for words and 6.50 [-7.65, -5.37] for characters. Character triples
also have worse log loss: 2.246 versus 2.171. The baseline failure is not just a
different choice of classification threshold.

## Interpretation and limitations

- **Supported:** in these reduced lexical spaces, other texts from the same known
  prompt contain useful information. The shared and independent Gaussian use the
  same fitted parameters. Group shuffling retains the inference compute and source
  composition but removes the benefit. Extra parameters or covariance tuning cannot
  explain this particular contrast.
- **Likely bottleneck:** the representation. Full-feature logistic regression beats
  reduced-feature logistic regression by 13.19 word-feature points and 11.04
  character-feature points. The current reduction/classifier pipeline loses much
  useful author signal. This does not establish that any chosen pretrained encoder
  will fix the problem.
- **Do not overinterpret estimated prompt variance:** the model attributes about
  92% of reduced word-feature variance and 85% of reduced character-feature variance
  to prompts. SVD favors high-variance directions, which may select prompt content.
  These fitted fractions are not an independent causal measurement.
- **Oracle groups only:** no unknown-prompt recovery was tested. Randomly joining
  unrelated passages harms accuracy and log loss. This is evidence that group errors
  matter, not a calibrated estimate of tolerance to realistic clustering errors.
- **Dataset structure is favorable:** 239 prompts have 18 authors, 43 have 12, with
  no repeated author within a prompt. Both sizes divide exactly into pairs/triples,
  so these real-data runs have no singleton remainders. Singleton inference and
  repeated authors are checked analytically and synthetically, not established on
  representative real batches.
- **Source robustness is incomplete:** the dataset includes 231 `new`, 8 `new2`, and
  43 `proxy_v1` prompts. On word features, pairs on `proxy_v1` lose 0.78 points to
  independence. On characters, the triple gain on `new2` is only 0.46 points.
  Within-source scores average recall over available classes (12 for `proxy_v1`,
  18 otherwise), so their absolute values are not directly comparable.
- **The shuffle control is conservative and imperfect:** 2.3–6.7% of its passage
  pairs still share a true prompt. Some folds have only one `new2` prompt, so
  within-source shuffling cannot break that prompt. The control does not remove all
  shared effects.
- **Uncertainty is conditional:** the bootstrap resamples prompts, not individual
  passages or partition repetitions. It conditions on fitted models and does not
  model training-set variation or correct for multiple comparisons. These are
  development results, not confirmation on an untouched test set.
- **Novelty is not established:** this does not reproduce or beat the earlier
  shared-Gaussian competition method. It is a controlled initial implementation.

## Execution and checks

The first one-fold smoke run reached 100 iterations with relative parameter change
4.14e-5, above the fixed 1e-5 tolerance. The limit was raised to 500 based on this
training convergence diagnostic. Features, shrinkage, seeds, and decision criteria
were unchanged. Full-run fits converged in 381–423 iterations for words and 272–301
for characters.

The word run took 84 seconds and the character run 120 seconds, running concurrently
with two numerical threads per process on this machine. These are end-to-end wall
times including baselines and bootstrap summaries, not portable performance claims.

The 11 automated checks pass. They cover direct joint-Gaussian probability agreement,
zero prompt covariance, permutation consistency, repeated author labels, invalid
partitions, synthetic parameter recovery and shuffle control, source-matched
fragmentation, prompt leakage rejection, embedding ID alignment, and bootstrap
aggregation. Ruff passes for all new code and tests. The original assignment
training code was not modified or rerun.

Reproduce using the two full-run commands in [README.md](README.md), with new output
directories. The downloaded version-7 data and input hash are recorded in each
manifest. Data, vectors, passage predictions, and environment files are not committed.
