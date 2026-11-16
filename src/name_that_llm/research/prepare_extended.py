"""Use eligible incomplete prompt blocks as well as complete blocks in version 13."""

import argparse
import json
from pathlib import Path

import pandas as pd

from name_that_llm.data import assign_folds
from name_that_llm.research.groups import validate_design
from name_that_llm.research.run import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/extended-v13/passages.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/extended-v13-expanded"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists.")
    raw = pd.read_csv(args.source, dtype=str, keep_default_na=False)
    eligible = raw.split.isin(["new", "new2", "proxy_v1"]) & raw.passes_funnel.str.lower().eq(
        "true"
    )
    frame = raw.loc[eligible, ["id", "prompt_id", "model", "text", "split"]].copy()
    if frame.duplicated(["prompt_id", "model"]).any():
        raise ValueError("Multiple attempts per author/prompt require explicit selection.")
    frame = assign_folds(frame)
    validate_design(frame)
    args.output.mkdir(parents=True)
    frame.to_csv(args.output / "folds.csv", index=False)
    report = dict(
        source_sha256=digest(args.source),
        raw_rows=len(raw),
        rows=len(frame),
        prompts=frame.prompt_id.nunique(),
        counts=frame.groupby("split").size().to_dict(),
        selection="passes_funnel; new/new2/proxy_v1; incomplete blocks allowed",
        exclusions="reconstructed train prompts, one-author reruns, pilot protocol",
    )
    (args.output / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
