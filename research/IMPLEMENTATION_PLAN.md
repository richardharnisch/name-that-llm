# Supervised-backbone integration plan

## Decision and scope

The current bottleneck is author information in the representation, not the ability
to compute prompt-group probabilities. Frozen MiniLM loses to character features.
That result does not test whether prompt correction adds value after supervised
attribution training. The next experiment must hold the trained backbone fixed
while comparing its direct head, full/reduced-vector logistic classifiers, LDA,
shared Gaussian, identical-parameter independence, and shuffled groups.

Do not add clustering, hard family routing, or more frozen-backbone sweeps yet.
The BPE vocabulary change is separate: it affects the assignment classifiers, not
the tokenizer of the pretrained encoder. It is not an attribution result.

## Responsibilities and implementation sequence

1. Training module and its tests: implementation agent `trainer`, GPT-6.1 Sol.
   Train only on an outer fold's training rows; fixed epoch budget, no outer-fold
   checkpoint selection. Export final vectors and direct-head predictions.
2. Artifact contract and adversarial tests: implementation agent `artifacts`,
   GPT-6.1 Sol. Verify fold, dataset hash, train/validation ID partitions, file
   hashes, and prediction consistency before accepting supervised vectors.
3. Integration, experiment design, review, documentation and commits: lead agent.
   Preserve the frozen-vector workflow. Add per-fold artifact loading and include
   the direct supervised head in comparisons and the advance criterion.

File ownership is disjoint. Agents do not commit or launch large training runs.
The lead reviews their changes, runs the combined tests and executes the pilot.

## Experimental contract

Use expanded version-13 data (9,482 rows, 660 prompts), existing prompt-disjoint
folds, and the cached MiniLM revision. Fixed three epochs, balanced author loss,
256-token truncation. Begin with fold 0 as a feasibility/development pilot, not a
five-fold performance claim. Report optimizer progress and truncation. Compare
against the earlier frozen and character runs on that exact fold.

An interrupted run must not pass as completed training. Every fold must receive
its own trained encoder. A single global supervised embedding archive is invalid
for cross-validation. The loader checks recorded provenance; it cannot establish
that an external producer truthfully recorded its training history.

The pretrained encoder may learn author information on all outer-training rows.
Logistic C selection within those rows is therefore not a fully nested supervised
representation experiment. The untouched outer validation remains the score being
reported. Claims about tuning robustness require inner-fold encoder retraining.
Gaussian moments estimated from supervised training embeddings can also differ
from their distribution on unseen prompts; report that limitation.

## Checks and stopping decision

Check training updates on a tiny local model; reject wrong-fold artifacts, overlapping
or missing IDs, modified files, and inconsistent predictions. Verify the supervised
head can block the experiment gate. Run all existing tests and lint changed modules.

One fold cannot satisfy the cross-validation gate. Advance to a larger run only if
training learns a useful author representation and prompt correction remains useful
against the direct head and full-vector classifier. No full-CV commitment follows
from a pipeline smoke test. Keep family diagnostics descriptive until a controlled
hierarchical-objective comparison is justified. A null result should stop expansion
of prompt machinery, not trigger validation-set hyperparameter search.
