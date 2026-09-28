"""Validate the released three-model journal and its grouped artifact files."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def main() -> int:
    manifest = json.loads((ARTIFACTS / "release_manifest.json").read_text(encoding="utf-8"))
    journal_paths = sorted((ARTIFACTS / "journal").glob("records-*.jsonl"))
    declared_shards = manifest.get("journal_shards", [])
    if [path.name for path in journal_paths] != [Path(item["file"]).name for item in declared_shards]:
        raise SystemExit("Journal shard inventory does not match release_manifest.json.")
    digest = hashlib.sha256()
    records = []
    for path, declaration in zip(journal_paths, declared_shards):
        payload = path.read_bytes()
        if len(payload) != declaration["bytes"] or hashlib.sha256(payload).hexdigest() != declaration["sha256"]:
            raise SystemExit(f"Journal shard hash or size mismatch: {path.name}.")
        digest.update(payload)
        records.extend(
            json.loads(line)
            for line in payload.decode("utf-8").splitlines()
            if line.strip()
        )
    digest = digest.hexdigest()
    if digest != manifest["selected_journal_sha256"]:
        raise SystemExit("Combined journal SHA-256 does not match release_manifest.json.")
    models = manifest["model_order"]
    datasets = manifest["dataset_counts"]
    pipelines = manifest["pipelines"]
    conditions = manifest["shot_conditions"]
    keys = [
        (
            row["phase"], row["model"], row["dataset"], row["example_id"],
            row["pipeline"], row["repetition"], row["shot_condition"],
        )
        for row in records
    ]
    if len(records) != manifest["observations"] or len(set(keys)) != len(keys):
        raise SystemExit("Journal count or uniqueness check failed.")
    if {row["model"] for row in records} != set(models):
        raise SystemExit("Journal contains a model outside the declared release scope.")
    if {row["dataset"] for row in records} != set(datasets):
        raise SystemExit("Journal datasets do not match the release manifest.")
    if any(row["pipeline"] not in pipelines or row["shot_condition"] not in conditions for row in records):
        raise SystemExit("Journal contains an undeclared pipeline or prompt condition.")

    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in records:
        grouped.setdefault((row["dataset"], row["pipeline"], row["model"]), []).append(row)
    expected_views = len(models) * len(datasets) * len(pipelines)
    expected_cells = expected_views * len(conditions)
    if len(grouped) != expected_views or manifest["complete_cells"] != expected_cells:
        raise SystemExit(
            f"Expected {expected_views} grouped views and {expected_cells} condition cells; "
            f"found {len(grouped)} grouped views."
        )

    for (dataset, pipeline, model), expected in grouped.items():
        pdir = {"direct": "p1", "code": "p2", "formulation_code": "p3"}[pipeline]
        mdir = model.replace(":", "_")
        directory = ARTIFACTS / dataset / pdir / mdir
        payload = json.loads((directory / "artifacts.json").read_text(encoding="utf-8"))
        stored = payload.get("records", [])
        expected_keys = {
            (row["phase"], row["example_id"], row["shot_condition"], row["repetition"])
            for row in expected
        }
        stored_keys = {
            (row["phase"], row["example_id"], row["shot_condition"], row["repetition"])
            for row in stored
        }
        if len(stored) != 2 * datasets[dataset] or stored_keys != expected_keys:
            raise SystemExit(f"Grouped artifact mismatch: {dataset}/{pdir}/{mdir}.")
        for condition in conditions:
            if sum(row["shot_condition"] == condition for row in stored) != datasets[dataset]:
                raise SystemExit(f"Condition denominator mismatch: {dataset}/{pdir}/{mdir}/{condition}.")
        if not (directory / "summary.md").is_file():
            raise SystemExit(f"Missing summary: {dataset}/{pdir}/{mdir}.")

    model_counts = Counter(row["model"] for row in records)
    print(
        f"Validated {len(records)} unique observations, {expected_cells} condition cells, "
        f"{len(grouped)} grouped views, and {len(models)} models. Journal SHA-256: {digest}"
    )
    for model in models:
        print(f"  {model}: {model_counts[model]} observations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
