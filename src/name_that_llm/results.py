"""Compare cross-validated runs with a Bayesian correlated t-test and a ROPE.

Benavoli et al. (2017), "Time for a change: a tutorial for comparing multiple classifiers
through Bayesian analysis". For two runs A and B on the same folds, the per-fold score
differences give a posterior over the true mean difference. The ROPE [-r, r] is the range of
differences that we treat as practically equivalent.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

SCORE_CHOICES = ("validation_accuracy", "validation_macro_f1", "validation_log_loss")


@dataclass(frozen=True)
class RopeResult:
    a: str
    b: str
    mean_difference: float  # mean of (score_a - score_b) over folds
    p_a_better: float  # P(difference > r)
    p_rope: float  # P(-r <= difference <= r)
    p_b_better: float  # P(difference < -r)


def load_scores(path: Path, score: str) -> tuple[str, np.ndarray]:
    """Read the per-fold scores of one run from its cv_metrics.json (or its directory)."""
    path = path / "cv_metrics.json" if path.is_dir() else path
    summary = json.loads(path.read_text(encoding="utf-8"))
    entry = summary["scores"].get(score)
    if entry is None:
        raise ValueError(f"{path} has no values for {score}.")
    name = f"{summary['model']}+{summary['encoder']}"
    return name, np.asarray(entry["values"], dtype=float)


def rope_probabilities(a: np.ndarray, b: np.ndarray, rope: float) -> tuple[float, float, float]:
    """Return P(A better), P(equivalent), P(B better) from the correlated t-test posterior.

    The folds share training data, so their differences are correlated. The posterior of the
    mean difference is a Student t with n-1 degrees of freedom, located at the sample mean,
    with scale sqrt(var * (1/n + rho / (1 - rho))) and rho = 1/n (test size / total size).
    """
    if len(a) != len(b):
        raise ValueError("Both runs need the same number of folds.")
    n = len(a)
    if n < 2:
        raise ValueError("At least two folds are needed.")
    differences = a - b
    mean = differences.mean()
    variance = differences.var(ddof=1)
    rho = 1 / n
    scale = np.sqrt(variance * (1 / n + rho / (1 - rho)))
    if scale == 0:  # identical differences on every fold: the posterior is a point mass
        return (float(mean > rope), float(abs(mean) <= rope), float(mean < -rope))
    posterior = stats.t(df=n - 1, loc=mean, scale=scale)
    p_b_better = posterior.cdf(-rope)
    p_a_better = posterior.sf(rope)
    return float(p_a_better), float(1 - p_a_better - p_b_better), float(p_b_better)


def compare(runs: dict[str, np.ndarray], rope: float) -> list[RopeResult]:
    """Compare every pair of runs."""
    results = []
    for (name_a, a), (name_b, b) in combinations(runs.items(), 2):
        p_a, p_rope, p_b = rope_probabilities(a, b, rope)
        results.append(RopeResult(name_a, name_b, float((a - b).mean()), p_a, p_rope, p_b))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        type=Path,
        nargs="+",
        help="cv_metrics.json files, or the output directories that contain them",
    )
    parser.add_argument("--score", choices=SCORE_CHOICES, default="validation_accuracy")
    parser.add_argument(
        "--rope",
        type=float,
        default=0.01,
        help="Half-width r of the ROPE [-r, r], in score units (0.01 = one accuracy point)",
    )
    parser.add_argument("--output", type=Path, help="Optionally save the results as JSON")
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error("Give at least two runs to compare.")
    if args.rope < 0:
        parser.error("--rope must not be negative.")

    try:
        runs = {}
        for path in args.runs:
            name, values = load_scores(path, args.score)
            if name in runs:
                name = f"{name} ({path.name if path.is_dir() else path.parent.name})"
            runs[name] = values
        results = compare(runs, args.rope)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Comparison failed: {error}\n")

    print(f"Score: {args.score}; ROPE: [-{args.rope}, {args.rope}]; folds: {len(next(iter(runs.values())))}")
    for name, values in runs.items():
        print(f"  {name:40s} mean {values.mean():.4f}  std {values.std(ddof=1):.4f}")
    print(f"\n{'A':32s} {'B':32s} {'mean A-B':>9s} {'P(A>B)':>8s} {'P(ROPE)':>8s} {'P(B>A)':>8s}")
    for r in results:
        print(
            f"{r.a:32s} {r.b:32s} {r.mean_difference:+9.4f} "
            f"{r.p_a_better:8.3f} {r.p_rope:8.3f} {r.p_b_better:8.3f}"
        )
    if args.output:
        payload = {"score": args.score, "rope": args.rope, "results": [asdict(r) for r in results]}
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
