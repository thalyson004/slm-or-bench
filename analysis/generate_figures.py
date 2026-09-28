"""Generate model-by-pipeline accuracy plots from summary.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from validate_primary import validate_primary_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("analysis/generated/summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("analysis/generated/accuracy_by_pipeline.pdf"))
    parser.add_argument("--phase", default="P1")
    parser.add_argument("--shot-condition", choices=("zero_shot", "few_shot"), default="zero_shot")
    args = parser.parse_args()
    with args.summary.open(encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    if args.phase == "P1":
        validate_primary_rows(all_rows, Path(__file__).resolve().parents[1])
    rows = [
        row for row in all_rows
        if row["phase"] == args.phase
        and row.get("shot_condition", "zero_shot") == args.shot_condition
    ]
    if not rows:
        raise SystemExit(f"No rows found for phase {args.phase}.")
    models = list(dict.fromkeys(row["model"] for row in rows))
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    pipelines = ["direct", "code", "formulation_code"]
    width = 0.24
    figure, axes = plt.subplots(
        len(datasets), 1, figsize=(max(9, len(models) * 1.1), 4 * len(datasets)), squeeze=False
    )
    for axis, dataset in zip(axes[:, 0], datasets):
        for offset, pipeline in enumerate(pipelines):
            values = []
            for model in models:
                selected = [
                    row for row in rows
                    if row["model"] == model
                    and row["pipeline"] == pipeline
                    and row["dataset"] == dataset
                ]
                total = sum(int(row["n"]) for row in selected)
                correct = sum(int(row["correct"]) for row in selected)
                values.append(correct / total if total else 0)
            positions = [index + (offset - 1) * width for index in range(len(models))]
            axis.bar(positions, values, width=width, label=pipeline)
        axis.set_title(f"{dataset} ({args.shot_condition.replace('_', '-')})")
        axis.set_ylabel("Strict objective accuracy")
        axis.set_ylim(0, 1)
        axis.set_xticks(range(len(models)), models, rotation=45, ha="right")
        axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
