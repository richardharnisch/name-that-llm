"""The direct fine-tuned head must remain a baseline in the research gate."""

import pandas as pd

from name_that_llm.research.evaluation import summarize


def test_supervised_head_is_compared_and_can_block_gate():
    rows = []
    methods = [
        "independent",
        "logistic_full",
        "logistic_dense",
        "lda",
        "shared_2",
        "shared_3",
        "centered_2",
        "centered_3",
        "shuffled_2",
        "shuffled_3",
        "finetuned_head",
    ]
    for prompt in range(8):
        for label in ["a", "b"]:
            for method in methods:
                rows.append(
                    dict(
                        id=f"{prompt}-{label}",
                        prompt_id=str(prompt),
                        model=label,
                        method=method,
                        fold=prompt % 2,
                        split="new",
                        nll=1.0,
                        correct=float(method.startswith("shared") or method == "finetuned_head"),
                    )
                )
    report = summarize(pd.DataFrame(rows), 100, 42)
    assert any(c["baseline"] == "finetuned_head" for c in report["comparisons"])
    for size in [2, 3]:
        criteria = report["initial_gate_criteria"][f"shared_{size}"]
        assert criteria["gain_at_least_one_point"]
        assert not criteria["beats_all_standalone_baselines"]
