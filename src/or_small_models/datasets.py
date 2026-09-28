"""Adapters for the three benchmark datasets used by the paper."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .records import DatasetExample

SUPPORTED_DATASETS = ("ComplexOR", "LogiOR", "IndustryOR")


def load_json_values(path: Path) -> list[dict[str, Any]]:
    """Load either a JSON array or concatenated top-level JSON objects."""
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list):
        return _dict_records(payload, path)
    if isinstance(payload, dict):
        return [payload]

    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        value, end = decoder.raw_decode(text, index)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at offset {index} in {path}.")
        records.append(value)
        index = end
    return records


def _dict_records(values: Iterable[Any], path: Path) -> list[dict[str, Any]]:
    records = list(values)
    if not all(isinstance(value, dict) for value in records):
        raise ValueError(f"Every record in {path} must be a JSON object.")
    return records


def _finite_float(value: Any, *, field: str, example_id: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{example_id}: {field} is not numeric.") from None
    if not math.isfinite(result):
        raise ValueError(f"{example_id}: {field} is not finite.")
    return result


def _structured_examples(dataset: str, path: Path) -> list[DatasetExample]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    examples = []
    for index, record in enumerate(payload.get("examples", [])):
        source = record.get("source", {})
        example_id = str(record.get("example_id", index))
        question = source.get("cleaned_question") or source.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{dataset}/{example_id}: missing verified question.")
        objective = _finite_float(
            record.get("canonical_expected_objective"),
            field="canonical_expected_objective",
            example_id=example_id,
        )
        canonical = record.get("canonical_model", {})
        examples.append(
            DatasetExample(
                dataset=dataset,
                example_id=example_id,
                question=question.strip(),
                expected_objective=objective,
                difficulty=(
                    str(source["difficulty"]).strip()
                    if source.get("difficulty") is not None
                    else None
                ),
                problem_type=(
                    str(canonical["family"]).strip()
                    if canonical.get("family") is not None
                    else None
                ),
            )
        )
    return examples


def load_dataset(dataset: str, dataset_root: Path) -> list[DatasetExample]:
    """Load the self-contained canonical release for one dataset."""
    if dataset in {"ComplexOR", "IndustryOR"}:
        return _structured_examples(dataset, dataset_root / f"{dataset}.json")
    if dataset == "LogiOR":
        records = load_json_values(dataset_root / "LogiOR.json")
        examples = []
        for index, record in enumerate(records):
            example_id = str(record.get("prob_id", index))
            question = record.get("description")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"LogiOR/{example_id}: missing description.")
            examples.append(
                DatasetExample(
                    dataset=dataset,
                    example_id=example_id,
                    question=question.strip(),
                    expected_objective=_finite_float(
                        record.get("expected_objective"),
                        field="expected_objective",
                        example_id=example_id,
                    ),
                    problem_type=(
                        str(record["problem_type"]).strip()
                        if record.get("problem_type") is not None
                        else None
                    ),
                )
            )
        return examples
    raise ValueError(
        f"Unsupported dataset '{dataset}'. Choose from: {', '.join(SUPPORTED_DATASETS)}."
    )


def select_examples(
    examples: list[DatasetExample],
    *,
    limit: int | None = None,
    example_ids: set[str] | None = None,
) -> list[DatasetExample]:
    if limit is not None and example_ids:
        raise ValueError("limit and example_ids are mutually exclusive.")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        return examples[:limit]
    if example_ids:
        selected = [item for item in examples if item.example_id in example_ids]
        missing = example_ids - {item.example_id for item in selected}
        if missing:
            raise ValueError(f"Unknown example IDs: {', '.join(sorted(missing))}.")
        return selected
    return examples


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
