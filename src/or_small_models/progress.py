"""Terminal progress display for resumable benchmark runs."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import Any, TextIO


class ProgressDisplay:
    """Render aggregate and current-cell progress from durable checkpoint keys.

    A TTY is rendered in place so a multi-day run occupies a compact block. When
    stdout is redirected, one concise line is emitted per completed observation.
    In both modes the counts are derived from ``ResultStore.completed``; resumed
    runs therefore start at the persisted checkpoint rather than at zero.
    """

    def __init__(
        self,
        *,
        phase: str,
        model_order: Iterable[str],
        dataset_order: Iterable[str],
        pipeline_order: Iterable[str],
        shot_condition_order: Iterable[str],
        selected_counts: dict[str, int],
        repetitions: int,
        stream: TextIO | None = None,
    ) -> None:
        self.phase = phase
        self.models = list(model_order)
        self.datasets = list(dataset_order)
        self.pipelines = list(pipeline_order)
        self.shots = list(shot_condition_order)
        self.selected_counts = dict(selected_counts)
        self.repetitions = repetitions
        self.stream = stream or sys.stdout
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._rendered_lines = 0

        per_example = len(self.pipelines) * len(self.shots) * repetitions
        self.expected_by_model = {
            model: sum(self.selected_counts.values()) * per_example
            for model in self.models
        }
        self.expected_by_dataset = {
            dataset: len(self.models) * count * per_example
            for dataset, count in self.selected_counts.items()
        }
        total_examples = sum(self.selected_counts.values())
        self.expected_by_pipeline = {
            pipeline: len(self.models) * total_examples * len(self.shots) * repetitions
            for pipeline in self.pipelines
        }
        self.expected_by_shot = {
            shot: len(self.models) * total_examples * len(self.pipelines) * repetitions
            for shot in self.shots
        }
        self.total = sum(self.expected_by_model.values())

    def start(self, completed: set[tuple[Any, ...]]) -> None:
        self.update(completed, status="running")

    def update(
        self,
        completed: set[tuple[Any, ...]],
        *,
        current: dict[str, Any] | None = None,
        status: str = "running",
    ) -> None:
        lines = self._lines(completed, current=current, status=status)
        if self.is_tty:
            if self._rendered_lines:
                self.stream.write(f"\x1b[{self._rendered_lines}A")
            for line in lines:
                self.stream.write("\x1b[2K\r" + line + "\n")
            self.stream.flush()
            self._rendered_lines = len(lines)
        else:
            # Redirected logs remain readable without emitting the full matrix.
            self.stream.write(lines[0] + " | " + lines[-1] + "\n")
            self.stream.flush()

    def finish(
        self,
        completed: set[tuple[Any, ...]],
        *,
        status: str,
        current: dict[str, Any] | None = None,
    ) -> None:
        self.update(completed, current=current, status=status)
        if self.is_tty:
            self.stream.write("\n")
            self.stream.flush()

    def _lines(
        self,
        completed: set[tuple[Any, ...]],
        *,
        current: dict[str, Any] | None,
        status: str,
    ) -> list[str]:
        total_done = self._count(completed)
        lines = [
            f"Benchmark {self.phase} | status={status} | "
            f"{self._progress(total_done, self.total)}",
        ]
        lines.extend(
            f"  model {model}: {self._progress(self._count(completed, model=model), self.expected_by_model[model])}"
            for model in self.models
        )
        lines.extend(
            f"  dataset {dataset}: {self._progress(self._count(completed, dataset=dataset), self.expected_by_dataset[dataset])}"
            for dataset in self.datasets
        )
        lines.extend(
            f"  pipeline {pipeline}: {self._progress(self._count(completed, pipeline=pipeline), self.expected_by_pipeline[pipeline])}"
            for pipeline in self.pipelines
        )
        lines.extend(
            f"  prompt {shot}: {self._progress(self._count(completed, shot=shot), self.expected_by_shot[shot])}"
            for shot in self.shots
        )
        if current:
            model = str(current.get("model", ""))
            dataset = str(current.get("dataset", ""))
            pipeline = str(current.get("pipeline", ""))
            shot = str(current.get("shot_condition", ""))
            cell_expected = self.selected_counts.get(dataset, 0) * self.repetitions
            cell_done = self._count(
                completed, model=model, dataset=dataset, pipeline=pipeline, shot=shot
            )
            lines.append(
                "Current  "
                f"{model} | {dataset} | {pipeline} | {shot} | "
                f"example={current.get('example_id', '')} | "
                f"cell {self._progress(cell_done, cell_expected)}"
            )
        else:
            lines.append("Current  waiting for next observation")
        return lines

    def _count(
        self,
        completed: set[tuple[Any, ...]],
        *,
        model: str | None = None,
        dataset: str | None = None,
        pipeline: str | None = None,
        shot: str | None = None,
    ) -> int:
        count = 0
        for key in completed:
            if len(key) < 7 or str(key[0]) != self.phase:
                continue
            if model is not None and str(key[1]) != model:
                continue
            if dataset is not None and str(key[2]) != dataset:
                continue
            if pipeline is not None and str(key[4]) != pipeline:
                continue
            if shot is not None and str(key[6]) != shot:
                continue
            count += 1
        return count

    @staticmethod
    def _progress(done: int, total: int, width: int = 20) -> str:
        bounded_total = max(total, 0)
        bounded_done = min(max(done, 0), bounded_total) if bounded_total else 0
        fraction = bounded_done / bounded_total if bounded_total else 1.0
        filled = round(width * fraction)
        percent = fraction * 100
        bar = "#" * filled + "-" * (width - filled)
        return f"[{bar}] {bounded_done}/{bounded_total} ({percent:5.1f}%)"
