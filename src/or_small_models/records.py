"""Typed records shared by the benchmark components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Provider = Literal["ollama", "openrouter"]
PipelineName = Literal["direct", "code", "formulation_code"]
ShotCondition = Literal["zero_shot", "few_shot"]


@dataclass(frozen=True)
class DatasetExample:
    dataset: str
    example_id: str
    question: str
    expected_objective: float
    difficulty: str | None = None
    problem_type: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: Provider
    size_b: float | None
    family: str
    specialization: str


@dataclass
class ModelResponse:
    content: str
    model: str
    provider: Provider
    latency_seconds: float
    request_attempts: int = 1
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    outcome: str
    success: bool
    objective_value: float | None = None
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    latency_seconds: float = 0.0


@dataclass
class BenchmarkRecord:
    run_id: str
    phase: str
    repetition: int
    dataset: str
    example_id: str
    difficulty: str | None
    problem_type: str | None
    model: str
    provider: Provider
    pipeline: PipelineName
    expected_objective: float
    predicted_objective: float | None
    correct: bool
    terminal_outcome: str
    stages: list[dict[str, Any]]
    total_latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    shot_condition: ShotCondition = "zero_shot"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
