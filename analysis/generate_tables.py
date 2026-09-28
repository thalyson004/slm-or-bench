"""Generate a compact CSV accuracy table from summary.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from validate_primary import validate_primary_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("analysis/generated/summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("analysis/generated/accuracy_table.csv"))
    parser.add_argument("--phase", default="P1")
    args = parser.parse_args()
    with args.summary.open(encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    if args.phase == "P1":
        validate_primary_rows(all_rows, Path(__file__).resolve().parents[1])
    rows = [row for row in all_rows if row["phase"] == args.phase]
    if not rows:
        raise SystemExit(f"No rows found for phase {args.phase}.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "provider", "dataset", "pipeline", "shot_condition", "correct", "n", "accuracy"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
