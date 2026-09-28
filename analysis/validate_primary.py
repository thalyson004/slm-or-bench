"""Refuse partial or duplicated P1 summaries before paper artifact generation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from or_small_models.datasets import load_dataset

PIPELINES = ("direct", "code", "formulation_code")
SHOT_CONDITIONS = ("zero_shot", "few_shot")


def validate_primary_rows(rows: list[dict[str, str]], repository: Path) -> None:
    model_payload = json.loads((repository / "configs" / "models.json").read_text(encoding="utf-8"))
    models = [item["id"] for item in model_payload["models"]]
    dataset_root = repository / "datasets"
    denominators = {
        dataset: len(load_dataset(dataset, dataset_root))
        for dataset in ("ComplexOR", "LogiOR", "IndustryOR")
    }
    primary = [row for row in rows if row["phase"] == "P1"]
    by_key = {}
    for row in primary:
        key = (
            row["model"], row["dataset"], row["pipeline"],
            row.get("shot_condition", "zero_shot"),
        )
        if key in by_key:
            raise ValueError(f"Duplicate P1 summary cell: {key}.")
        by_key[key] = row
    expected = {
        (model, dataset, pipeline, shot_condition)
        for model in models
        for dataset in denominators
        for pipeline in PIPELINES
        for shot_condition in SHOT_CONDITIONS
    }
    missing = sorted(expected - set(by_key))
    extra = sorted(set(by_key) - expected)
    wrong_n = [
        (key, int(row["n"]), denominators[key[1]])
        for key, row in by_key.items()
        if int(row["n"]) != denominators[key[1]]
    ]
    if missing or extra or wrong_n:
        raise ValueError(
            f"Incomplete P1 summary: missing={len(missing)}, extra={len(extra)}, "
            f"wrong_denominator={wrong_n[:10]}."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("analysis/generated/summary.csv"))
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    with args.summary.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    validate_primary_rows(rows, repository)
    model_count = len(
        json.loads((repository / "configs" / "models.json").read_text(encoding="utf-8"))["models"]
    )
    print(f"P1 summary is complete: {model_count} models x 3 datasets x 3 pipelines x 2 shot conditions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
