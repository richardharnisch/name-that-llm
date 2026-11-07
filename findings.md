# Findings

## What the assignment asks, and how we address it

The checklist in `nlp1-student/part1/project.ipynb` has four requirements.

| Requirement | How we address it | Where |
|---|---|---|
| Compare text classification systems on our data and task | 9 systems (3 models x 3 encoders) on 18-way LLM attribution | Section 1 |
| Cover the two tutorial representations plus one of our own | Token indicators and bag-of-words (tutorial), plus **tf-idf** | Section 1 |
| Cover the two tutorial architectures plus one of our own | Log-linear and one-hidden-layer MLP (tutorial), plus a **linear SVM** (same linear layer, trained with a multi-class hinge loss) | Section 1 |
| Cross-validation and a relevant metric | 5-fold CV with whole prompts kept in one fold. Validation accuracy is the main metric, with macro F1 and log loss also recorded | Setup |
| Bayesian estimation with a ROPE, with a size we motivate | Bayesian correlated t-test on per-fold accuracy differences, ROPE of +-0.01 | Section 2 |

Still open: the SVM is a modification of the log-linear model, not of the MLP, so a deeper or dropout MLP variant would be a closer match to the notebook's suggestion (see "Not done").

## Setup

- Task: 18-class classification of which LLM wrote a passage (chance is about 5.6%).
- Data: 4,818 passages from 282 prompts, in 5 cross-validation folds. Whole prompts stay in one fold, so validation prompts are unseen.
- Tokenizer: BPE with a vocabulary of 8000 and minimum frequency 2, trained per fold on the training passages only.
- Models: log-linear, MLP (one hidden layer of 256 units) and linear SVM.
- Encoders: bag-of-words, token-indicator and tf-idf, all L2-normalised. The tf-idf weights come from the training passages only.
- Training: Adam, learning rate 1e-3, weight decay 1e-4, batch size 64. Training stops when the training loss improves by less than 1% over 10 epochs (cap 200). Validation data never decides when to stop.
- Comparison: Bayesian correlated t-test with a ROPE (`src/name_that_llm/results.py`), on per-fold validation accuracy.

## 1. Baseline grid (3 models x 3 encoders)

Validation accuracy, mean ± std over 5 folds. Runs are in `artifacts/rope/`.

| Encoder | log-linear | MLP | SVM |
|---|---|---|---|
| bag-of-words | 0.294 ± 0.027 | 0.297 ± 0.021 | 0.291 ± 0.020 |
| token-indicator | **0.304 ± 0.027** | 0.281 ± 0.020 | 0.292 ± 0.022 |
| tf-idf | 0.284 ± 0.018 | 0.269 ± 0.021 | 0.274 ± 0.018 |

- Every run scores 27-30%, far above chance but low in absolute terms.
- The models memorise the training data: train accuracy is 0.89-1.00, and the gap to validation is about 0.6-0.7.
- Differences between runs are mostly within the fold-to-fold noise (std about 0.02).

## 2. ROPE analysis

### Why this is Bayesian

A cross-validation mean with a standard deviation cannot tell us whether two systems differ, and a classical significance test answers the wrong question. A frequentist t-test gives a p-value under the assumption that the two systems are exactly equal. It can reject that assumption, but it can never tell us that two systems are practically the same, because failing to reject is not evidence of equality. With only 5 folds it also has little power.

The Bayesian approach (Benavoli et al., 2017) instead puts a posterior distribution on the true mean accuracy difference between two systems A and B, given the per-fold differences. From that posterior we can read off three probabilities directly:

- **P(A better):** the probability that A beats B by more than r.
- **P(ROPE):** the probability that the difference lies in [-r, r], so the systems are practically equivalent.
- **P(B better):** the probability that B beats A by more than r.

This is the statement we actually want, and "practically equivalent" becomes something we can state with a probability rather than an absence of evidence.

### Method

- **ROPE (region of practical equivalence):** the interval [-r, r] of accuracy differences we treat as "the same model".
- **Paired differences:** for each pair of systems we take the per-fold difference in validation accuracy. All systems use the same 5 folds, so the pairing is valid.
- **Correlated t-test:** the folds share training passages, so their scores are correlated. The posterior is a Student t with n-1 degrees of freedom, centred on the mean difference, with scale sqrt(var * (1/n + rho/(1-rho))) and rho = 1/n (the Nadeau and Bengio correction). Ignoring rho would make the posterior too narrow and overstate how sure we are.
- **Choice of r = 0.01 (one accuracy point):** this is a choice, not a derived value. One point is about 48 of the 4,818 passages, which is a difference we would not call meaningful. It is also smaller than the fold-to-fold std of about 0.02, so it keeps the analysis from declaring everything equivalent. We use r = 0.05 as a sensitivity check.

Output is saved in `artifacts/rope/rope_acc.json` (r = 0.01). Command: `uv run python scripts/results.py artifacts/rope/<run> ... --rope <r>`.

### Results

- **r = 0.05:** all 36 pairs have P(ROPE) of at least 0.89. No model is distinguishable from any other. The largest mean gap is 3.5 points, so this ROPE is too wide to be informative.
- **r = 0.01:** most pairs are unresolved (P(ROPE) between 0.4 and 0.7, with a real chance of either model being better).

Selected pairs at r = 0.01 (A − B is the mean per-fold accuracy difference; P(A>B) means A is better by more than r):

| A | B | A − B | P(A>B) | P(ROPE) | P(B>A) |
|---|---|---|---|---|---|
| log-linear + token-indicator | MLP + tf-idf | +0.035 | 0.976 | 0.021 | 0.004 |
| log-linear + bag-of-words | MLP + tf-idf | +0.025 | 0.975 | 0.024 | 0.001 |
| MLP + bag-of-words | MLP + tf-idf | +0.027 | 0.942 | 0.052 | 0.006 |
| log-linear + token-indicator | MLP + token-indicator | +0.023 | 0.930 | 0.065 | 0.005 |
| SVM + bag-of-words | MLP + tf-idf | +0.021 | 0.904 | 0.089 | 0.006 |
| MLP + bag-of-words | SVM + token-indicator | +0.005 | 0.040 | 0.959 | 0.001 |
| SVM + bag-of-words | SVM + token-indicator | −0.001 | 0.114 | 0.739 | 0.147 |
| log-linear + bag-of-words | MLP + bag-of-words | −0.003 | 0.115 | 0.656 | 0.228 |
| log-linear + bag-of-words | log-linear + token-indicator | −0.011 | 0.014 | 0.453 | 0.532 |
| log-linear + token-indicator | MLP + bag-of-words | +0.008 | 0.379 | 0.597 | 0.024 |

The first five rows are the clearest differences, the next two are probably equivalent, and the last three are unresolved. All 36 pairs are in `artifacts/rope/rope_acc.json`.

- Clearest result at r = 0.01: MLP + tf-idf (0.269) is probably worse than log-linear + bag-of-words (P = 0.975), log-linear + token-indicator (P = 0.976) and MLP + bag-of-words (P = 0.942).
- No model is clearly the best. Log-linear + token-indicator has the highest mean, but its probability of beating most others is only 0.6-0.9.
- Conclusion: representation and architecture matter little here, apart from tf-idf with the MLP.

## 3. Weight decay (regularization)

Weight decay 1e-3, 1e-2 and 1e-1 was tried on log-linear + bag-of-words, log-linear + token-indicator and MLP + bag-of-words. Runs are in `artifacts/wd/`.

| Setup | wd 1e-4 | wd 1e-3 | wd 1e-2 | wd 1e-1 |
|---|---|---|---|---|
| log-linear + bag-of-words (val) | 0.294 | 0.243 | 0.184 | 0.126 |
| log-linear + token-indicator (val) | 0.304 | 0.281 | 0.231 | 0.125 |
| MLP + bag-of-words (val) | 0.297 | 0.304 | 0.112 | 0.059 |

- Stronger weight decay did not improve validation accuracy. The only run level with the baseline is the MLP at 1e-3 (0.304 against 0.297, inside the noise).
- The train/validation gap shrinks only because train accuracy falls. The models underfit.
- The MLP collapses to near chance at 1e-2 and above.
- Early stopping is not the cause. Rerunning 1e-3 and 1e-2 for a fixed 100 epochs with no early stopping (`artifacts/wd_fixed/`) changed train and validation accuracy by at most about 0.06, and no run improved.

## Interpretation

L2 regularization does not close the gap, so it is probably not plain overfitting. A likely explanation, which we have not tested, is that validation prompts are new topics. Topic words learned from training prompts then do not transfer. This would also fit tf-idf scoring slightly worse, since it up-weights rare, topic-specific tokens.

## Caveats

- 5 folds and one seed, so the uncertainty is large.
- The ROPE width is a choice that has to be justified. 0.01 gives a result worth discussing, while 0.05 makes everything equivalent.
- The weight-decay settings were chosen by looking at CV results. They are not a clean held-out estimate.
- The weight-decay runs are not included in the ROPE comparison.
- `data/folds.csv` and the per-fold tokenizers had to be regenerated (`scripts/split_data.py`, `scripts/preprocess_data.py`) before training, because the existing `data/` was stale.

## Not done

- A modification of the MLP (more layers, dropout, other hidden sizes) as the extra architecture. The SVM is our extra architecture for now.
- More data: the split excludes the competition prompts, the pilot and the Kimi-only rerun on purpose. Adding them would change the task.
- MLP dropout, a smaller vocabulary and other regularizers.
- The notebook also asks for a record of AI assistance, the sources used, and backup slides for the oral. These are not part of this file.

## Sources

- Benavoli, Corani, Demšar and Zaffalon (2017). Time for a change: a tutorial for comparing multiple classifiers through Bayesian analysis. Journal of Machine Learning Research 18.
- Nadeau and Bengio (2003). Inference for the generalization error. Machine Learning 52. This is the source of the rho = 1/n variance correction.
