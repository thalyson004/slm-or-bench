"""Re-execute generated code evaluated by an obsolete execution filter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from or_small_models.artifacts import ResultStore  # noqa: E402
from or_small_models.executor import GurobiExecutor  # noqa: E402
from or_small_models.extraction import objectives_match  # noqa: E402

LEGACY_OUTCOME = "policy_error"
LEGACY_REEVALUATION_TYPE = "policy_error_reexecution"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_dir",
        nargs="?",
        type=Path,
        default=REPOSITORY / "results",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Atomically replace eligible evaluations and rebuild affected cells.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result_dir = args.result_dir.resolve()
    state_dir = result_dir / "_state"
    ledger_path = state_dir / "records.jsonl"
    manifest_path = state_dir / "manifest.json"
    progress_path = state_dir / "progress.json"

    progress = _read_json(progress_path)
    if progress.get("status") == "running":
        raise RuntimeError(
            "The benchmark is running. Interrupt it safely before re-evaluating records."
        )
    records = _read_jsonl(ledger_path)
    candidates = [
        index for index, record in enumerate(records) if _needs_reevaluation(record)
    ]
    print(f"Generated-code records to re-evaluate: {len(candidates)}")
    if not args.apply:
        print("No files changed. Pass --apply after safely interrupting the benchmark.")
        return 0

    manifest = _read_json(manifest_path)
    settings = manifest["frozen_plan"]["settings"]
    executor = GurobiExecutor(float(settings["code_timeout_seconds"]))
    source_stat = ledger_path.stat()
    changed_cells: set[tuple[str, str, str]] = set()
    outcomes: Counter[str] = Counter()

    for position, index in enumerate(candidates, 1):
        updated = reevaluate_record(
            records[index],
            executor=executor,
            absolute_tolerance=float(settings["absolute_tolerance"]),
            relative_tolerance=float(settings["relative_tolerance"]),
        )
        records[index] = updated
        changed_cells.add(
            (updated["dataset"], updated["pipeline"], updated["model"])
        )
        outcomes[str(updated["terminal_outcome"])] += 1
        print(
            f"[{position}/{len(candidates)}] {updated['model']} / "
            f"{updated['dataset']} / {updated['example_id']} / "
            f"{updated['pipeline']} / {updated['shot_condition']} -> "
            f"{updated['terminal_outcome']}"
        )

    _assert_ledger_unchanged(ledger_path, source_stat, progress_path)
    _write_jsonl_atomic(ledger_path, records)
    store = ResultStore(result_dir, manifest, record_resume_event=False)
    store.rebuild_cells(changed_cells)
    store.write_progress(
        status=str(progress.get("status", "interrupted")),
        current=progress.get("current_observation"),
        interruption=progress.get("interruption"),
    )
    legacy_audit = state_dir / "policy_reevaluation.json"
    legacy_audit.unlink(missing_ok=True)
    print(f"Updated checkpoint: {ledger_path}")
    print(f"Rebuilt result cells: {len(changed_cells)}")
    print(f"Final outcomes: {dict(sorted(outcomes.items()))}")
    return 0


def reevaluate_record(
    record: dict[str, Any],
    *,
    executor: Any,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    if not _needs_reevaluation(record):
        raise ValueError("Record was not produced by the obsolete execution filter.")
    updated = deepcopy(record)
    execution_index, previous_execution = _execution_stage(updated)
    result = executor.execute(str(previous_execution["code"]))
    updated["stages"][execution_index] = {
        "stage": "execution",
        "code": previous_execution["code"],
        **asdict(result),
    }
    predicted = result.objective_value
    correct = objectives_match(
        predicted,
        float(updated["expected_objective"]),
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    terminal = result.outcome
    if terminal == "executed":
        terminal = "correct" if correct else "incorrect_objective"
    updated["predicted_objective"] = predicted
    updated["correct"] = correct
    updated["terminal_outcome"] = terminal
    updated["total_latency_seconds"] = sum(
        float(stage.get("latency_seconds", 0.0)) for stage in updated["stages"]
    )
    updated.pop("reevaluations", None)
    return updated


def _needs_reevaluation(record: dict[str, Any]) -> bool:
    if record.get("terminal_outcome") == LEGACY_OUTCOME:
        return _has_generated_code(record)
    return any(
        item.get("type") == LEGACY_REEVALUATION_TYPE
        for item in record.get("reevaluations", [])
        if isinstance(item, dict)
    ) and _has_generated_code(record)


def _has_generated_code(record: dict[str, Any]) -> bool:
    try:
        _, execution = _execution_stage(record)
    except ValueError:
        return False
    return isinstance(execution.get("code"), str) and bool(execution["code"].strip())


def _execution_stage(record: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    for index, stage in enumerate(record.get("stages", [])):
        if stage.get("stage") == "execution":
            return index, stage
    raise ValueError("Record has no execution stage.")


def _assert_ledger_unchanged(
    ledger_path: Path, source_stat: os.stat_result, progress_path: Path
) -> None:
    current = ledger_path.stat()
    progress = _read_json(progress_path)
    if progress.get("status") == "running":
        raise RuntimeError("Benchmark resumed during re-evaluation; no files were changed.")
    if (
        current.st_size != source_stat.st_size
        or current.st_mtime_ns != source_stat.st_mtime_ns
    ):
        raise RuntimeError("Checkpoint changed during re-evaluation; no files were changed.")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
