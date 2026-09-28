"""The three frozen pipelines, with no semantic retry or self-correction."""

from __future__ import annotations

import time
from dataclasses import asdict
from .clients import ModelClient
from .errors import ModelRequestError
from .executor import GurobiExecutor
from .extraction import extract_direct_objective, extract_python_code, objectives_match
from .prompts import (
    SYSTEM,
    code_prompt,
    direct_prompt,
    formulation_code_prompt,
    formulation_prompt,
)
from .records import BenchmarkRecord, DatasetExample, ModelResponse, PipelineName, ShotCondition


def run_pipeline(
    *,
    run_id: str,
    phase: str,
    repetition: int,
    shot_condition: ShotCondition = "zero_shot",
    pipeline: PipelineName,
    example: DatasetExample,
    client: ModelClient,
    executor: GurobiExecutor,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> BenchmarkRecord:
    started = time.perf_counter()
    stages: list[dict] = []
    predicted: float | None = None
    terminal = "model_error"
    active_stage = "model_call"
    active_prompt = ""
    try:
        if pipeline == "direct":
            prompt = direct_prompt(example.question, shot_condition)
            active_stage = "direct_answer"
            active_prompt = prompt
            response = client.generate(SYSTEM, prompt)
            stages.append(_model_stage("direct_answer", prompt, response))
            if _is_truncated(response):
                terminal = "truncated_response"
            else:
                predicted = extract_direct_objective(response.content)
                terminal = "parsed" if predicted is not None else "parse_error"
        elif pipeline == "code":
            prompt = code_prompt(example.question, shot_condition)
            active_stage = "code_generation"
            active_prompt = prompt
            response = client.generate(SYSTEM, prompt)
            stages.append(_model_stage("code_generation", prompt, response))
            if _is_truncated(response):
                terminal = "truncated_response"
            else:
                predicted, terminal = _execute_response(response, executor, stages)
        elif pipeline == "formulation_code":
            first_prompt = formulation_prompt(example.question, shot_condition)
            active_stage = "formulation"
            active_prompt = first_prompt
            formulation = client.generate(SYSTEM, first_prompt)
            stages.append(_model_stage("formulation", first_prompt, formulation))
            if _is_truncated(formulation):
                terminal = "truncated_response"
            else:
                second_prompt = formulation_code_prompt(
                    example.question, formulation.content, shot_condition
                )
                active_stage = "code_from_formulation"
                active_prompt = second_prompt
                code = client.generate(SYSTEM, second_prompt)
                stages.append(_model_stage("code_from_formulation", second_prompt, code))
                if _is_truncated(code):
                    terminal = "truncated_response"
                else:
                    predicted, terminal = _execute_response(code, executor, stages)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")
    except ModelRequestError as error:
        stages.append(
            {
                "stage": active_stage,
                "system_prompt": SYSTEM,
                "user_prompt": active_prompt,
                **error.to_dict(),
            }
        )
        terminal = error.code
    except RuntimeError as error:
        stages.append(
            {
                "stage": active_stage,
                "system_prompt": SYSTEM,
                "user_prompt": active_prompt,
                "error_type": "model_error",
                "error": str(error),
            }
        )
        terminal = "model_error"
    except ValueError as error:
        stages.append({"stage": "response_extraction", "error": str(error)})
        terminal = "invalid_response"

    correct = objectives_match(
        predicted,
        example.expected_objective,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    if terminal in {"parsed", "executed"}:
        terminal = "correct" if correct else "incorrect_objective"
    prompt_tokens = _sum_optional(stages, "prompt_tokens")
    completion_tokens = _sum_optional(stages, "completion_tokens")
    return BenchmarkRecord(
        run_id=run_id,
        phase=phase,
        repetition=repetition,
        dataset=example.dataset,
        example_id=example.example_id,
        difficulty=example.difficulty,
        problem_type=example.problem_type,
        model=client.model,
        provider=client.provider,
        pipeline=pipeline,
        expected_objective=example.expected_objective,
        predicted_objective=predicted,
        correct=correct,
        terminal_outcome=terminal,
        stages=stages,
        total_latency_seconds=time.perf_counter() - started,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        shot_condition=shot_condition,
    )


def _execute_response(
    response: ModelResponse,
    executor: GurobiExecutor,
    stages: list[dict],
) -> tuple[float | None, str]:
    code = extract_python_code(response.content)
    result = executor.execute(code)
    stages.append({"stage": "execution", "code": code, **asdict(result)})
    return result.objective_value, result.outcome


def _model_stage(name: str, prompt: str, response: ModelResponse) -> dict:
    return {
        "stage": name,
        "system_prompt": SYSTEM,
        "user_prompt": prompt,
        "response": response.content,
        "resolved_model": response.model,
        "provider": response.provider,
        "latency_seconds": response.latency_seconds,
        "request_attempts": response.request_attempts,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "finish_reason": response.finish_reason,
        "provider_metadata": response.provider_metadata,
        "raw_response": response.raw_response,
    }


def _sum_optional(stages: list[dict], key: str) -> int | None:
    values = [stage[key] for stage in stages if isinstance(stage.get(key), int)]
    return sum(values) if values else None


def _is_truncated(response: ModelResponse) -> bool:
    return (response.finish_reason or "").casefold() in {"length", "max_tokens"}
