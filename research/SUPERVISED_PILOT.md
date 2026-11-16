# Supervised pilot: do not expand the prompt model yet

2026-11-16. Implementation: `8ec32c6`. The lead reviewed and integrated two GPT-6.1
Sol contributions with disjoint ownership. The integrated suite has 40 passing
checks. See [implementation plan](IMPLEMENTATION_PLAN.md) for decisions and commands.
Exact metrics and provenance are in [results/supervised-pilot.json](results/supervised-pilot.json).

## Same-fold comparison

Fixed three-epoch MiniLM author training, fold 0 only. Validation contains 1,886
passages from 132 unseen prompts. No validation checkpoint selection. Training
losses: 2.618, 2.174, 2.022; roughly 29 training passages/second on CPU. The encoder
and head both receive updates, with separate learning rates. Of the full dataset,
1,022 passages exceed the 256-token input limit. This remains a limitation.

| Method on fold 0 | Balanced accuracy | Log loss |
|---|---:|---:|
| Frozen MiniLM, full-vector logistic | 14.09% | — |
| Character TF-IDF, full-vector logistic | 30.52% | — |
| Supervised MiniLM direct head | 22.35% | 2.035 |
| Supervised MiniLM, full-vector logistic | **29.56%** | **1.885** |
| Supervised MiniLM, reduced-vector logistic | 28.09% | 1.911 |
| Matched independent Gaussian | 27.19% | 3.730 |
| Shared Gaussian, pairs | 27.59% | 3.707 |
| Shared Gaussian, triples | 27.52% | 3.695 |
| Centering + LDA, triples | 28.76% | 2.206 |
| Shared Gaussian, shuffled triples | 27.34% | 3.793 |

Pairs gain 0.40 percentage points over independence (90% paired prompt-bootstrap
interval -0.17 to +0.95). Triples gain 0.33 (-0.31 to +0.98). Both intervals include
zero and both point estimates miss the prespecified one-point threshold. Triples
lose 2.04 points to full-vector logistic regression (-3.35 to -0.66). Gaussian
fitting converged after 61 iterations. The experiment gate remains false; a
single-fold pilot could not pass the full-CV gate in any case.

## Research decision

Supervised attribution training materially improves the fixed encoder's useful
features over freezing it. However, the large prompt-correction gains seen with
frozen/lexical features do not persist in this pilot. This is the right failure
mode to learn about before investing in larger encoders or unknown-group recovery.

Do not launch the unchanged five-fold supervised experiment or add clustering on
this evidence. The next focused diagnostic is whether training-derived Gaussian
means and covariances transfer to unseen prompts. Its very poor log loss compared
with discriminative heads is consistent with distribution mismatch or probability
miscalibration, but does not identify the cause. Estimate that mismatch using an
inner prompt holdout, keeping outer validation out of model selection. Simple
probability calibration alone would not establish an accuracy benefit.

The fitted prompt covariance accounts for 11.8% of feature variance in this run.
That is compatible with a smaller prompt effect, not proof that author training
causally removed it. The model and feature space both changed.

This does not disprove all prompt modeling or establish a competitive backbone:
only one development fold and one compact encoder were tested. The direct head
has zero recall on five labels; its three-epoch budget is not an optimized
fine-tuning result. Full-vector logistic probes outperform it substantially.
Training a larger model before diagnosing these issues would change too many
factors at once.

## Family perspective

For the supervised full-vector logistic probe, exact-author recall is 29.56% while
provider agreement is 69.03%, averaging equally over authors. Errors divide into
15.46% within operational families, 24.01% between families within a provider,
and 30.97% across providers. These shares plus correct predictions sum to 100%.
Thus finer identification within a provider matters, but cross-provider errors
are also substantial. These are not measurements with the true family supplied.

Operational-family exact recall includes Claude Opus 36.80%, GPT-5.6 variants
15.52%, Gemini Flash 26.67%, and GLM-5.3 variants 16.02%. Group definitions are
name-based rather than verified model genealogy. Full per-author metrics remain
in the archived summary. Hard family routing is not justified by these numbers.

## Integration review

- Training and evaluation are separate modules. Each fold has its own encoder.
- The artifact loader checks fold and dataset hashes, exact training/validation
  partitions, completed training, vector/prediction hashes and prediction labels.
- Prompt/source metadata are verified against the dataset before bootstrap scoring.
- The direct trained head is a required competitor in the supervised gate.
- Existing frozen-vector and lexical experiment paths remain available.
- Test coverage includes local training updates, partial-run rejection, full mocked
  export, corrupted/mismatched artifacts, metadata errors and the supervised gate.
- All 40 tests and Ruff checks pass. The real training/export/load/evaluation/family
  reporting path also completed successfully. Large weights and data stay local.
