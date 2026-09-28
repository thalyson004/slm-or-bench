"""Normalize JSONL benchmark records and aggregate model/pipeline outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/generated"))
    return parser.parse_args()


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    seen = set()
    for path in paths:
        candidates = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
        for candidate in candidates:
            for number, line in enumerate(candidate.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSONL at {candidate}:{number}.") from None
                key = (
                    record.get("phase"), record.get("model"), record.get("dataset"),
                    record.get("example_id"), record.get("pipeline"), record.get("repetition"),
                    record.get("shot_condition", "zero_shot"),
                )
                if key in seen:
                    raise ValueError(f"Duplicate experimental key {key} at {candidate}:{number}.")
                seen.add(key)
                record["source_file"] = str(candidate)
                records.append(record)
    return records


def write_items(records: list[dict], path: Path) -> None:
    fields = [
        "run_id", "phase", "repetition", "shot_condition", "dataset", "example_id", "difficulty",
        "problem_type", "model", "provider", "pipeline", "expected_objective",
        "predicted_objective", "correct", "terminal_outcome", "total_latency_seconds",
        "prompt_tokens", "completion_tokens", "source_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_summary(records: list[dict], path: Path) -> None:
    groups = defaultdict(list)
    for record in records:
        groups[(record["phase"], record["model"], record["provider"], record["dataset"], record["pipeline"], record.get("shot_condition", "zero_shot"))].append(record)
    fields = [
        "phase", "model", "provider", "dataset", "pipeline", "shot_condition", "n", "correct", "accuracy",
        "completed_objective", "completion_rate", "median_latency_seconds",
        "prompt_tokens", "completion_tokens", "outcome_counts_json",
    ]
    rows = []
    for key, items in sorted(groups.items()):
        correct = sum(bool(item["correct"]) for item in items)
        completed = sum(item.get("predicted_objective") is not None for item in items)
        outcomes = defaultdict(int)
        for item in items:
            outcomes[item["terminal_outcome"]] += 1
        rows.append({
            "phase": key[0], "model": key[1], "provider": key[2], "dataset": key[3], "pipeline": key[4], "shot_condition": key[5],
            "n": len(items), "correct": correct, "accuracy": correct / len(items),
            "completed_objective": completed, "completion_rate": completed / len(items),
            "median_latency_seconds": statistics.median(float(item["total_latency_seconds"]) for item in items),
            "prompt_tokens": sum(item.get("prompt_tokens") or 0 for item in items),
            "completion_tokens": sum(item.get("completion_tokens") or 0 for item in items),
            "outcome_counts_json": json.dumps(dict(sorted(outcomes.items())), sort_keys=True),
        })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    records = load_records(args.inputs)
    if not records:
        raise SystemExit("No benchmark records found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_items(records, args.output_dir / "item_results.csv")
    write_summary(records, args.output_dir / "summary.csv")
    print(f"Normalized {len(records)} records into {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
