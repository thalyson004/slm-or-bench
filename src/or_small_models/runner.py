"""Checkpointed execution of the frozen benchmark matrix."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import ResultStore
from .clients import OllamaClient, OpenRouterClient, ollama_inventory, openrouter_inventory
from .datasets import file_sha256, load_dataset, select_examples
from .executor import GurobiExecutor
from .pipelines import run_pipeline
from .progress import ProgressDisplay
from .records import ModelSpec, PipelineName, ShotCondition


class BenchmarkInterrupted(RuntimeError):
    """Raised after durable state is finalized following a user interruption."""


class BenchmarkWaitingForCredits(RuntimeError):
    """Raised when an OpenRouter request cannot run because credits are unavailable."""


INFRASTRUCTURE_TERMINALS = {
    "authentication_error",
    "transport_error",
    "rate_limit_error",
    "provider_server_error",
    "provider_http_error",
    "invalid_retry_delay",
    "model_error",
    "insufficient_credits",
}


@dataclass(frozen=True)
class ExperimentSettings:
    temperature: float
    seed: int
    max_output_tokens: int
    request_timeout_seconds: float
    retry_request_timeouts: bool
    code_timeout_seconds: float
    max_transport_retries: int
    absolute_tolerance: float
    relative_tolerance: float


def load_settings(path: Path) -> ExperimentSettings:
    payload = json.loads(path.read_text(encoding="utf-8"))
    keys = ExperimentSettings.__dataclass_fields__.keys()
    return ExperimentSettings(**{key: payload[key] for key in keys})


def load_model_specs(path: Path) -> list[ModelSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ModelSpec(**item) for item in payload["models"]]


def select_models(specs: list[ModelSpec], requested: list[str]) -> list[ModelSpec]:
    if requested == ["all"]:
        return specs
    by_id = {spec.id: spec for spec in specs}
    missing = [model for model in requested if model not in by_id]
    if missing:
        raise ValueError(f"Unknown models: {', '.join(missing)}.")
    return [by_id[model] for model in dict.fromkeys(requested)]


def select_provider_models(
    specs: list[ModelSpec], provider: str
) -> list[ModelSpec]:
    if provider == "all":
        return specs
    if provider not in {"ollama", "openrouter"}:
        raise ValueError(f"Unknown provider selection: {provider}.")
    selected = [spec for spec in specs if spec.provider == provider]
    if not selected:
        raise ValueError(f"No selected model uses provider {provider}.")
    return selected


def run_benchmark(
    *,
    phase: str,
    specs: list[ModelSpec],
    datasets: list[str],
    pipelines: list[PipelineName],
    shot_conditions: list[ShotCondition],
    repetitions: int,
    repetition_start: int,
    dataset_root: Path,
    output_dir: Path,
    settings: ExperimentSettings,
    limit: int | None,
    example_ids_by_dataset: dict[str, set[str]] | None,
    ollama_base_url: str,
    openrouter_base_url: str,
    openrouter_api_key: str,
    execution_provider: str = "all",
) -> tuple[Path, Path]:
    if repetitions <= 0 or repetition_start <= 0:
        raise ValueError("repetitions and repetition_start must be positive.")
    if not shot_conditions:
        raise ValueError("At least one shot condition is required.")
    execution_specs = select_provider_models(specs, execution_provider)
    if (
        any(spec.provider == "openrouter" for spec in execution_specs)
        and not openrouter_api_key
    ):
        raise RuntimeError("OPENROUTER_API_KEY is required before the run starts.")

    inventory = _ollama_inventory_for_plan(
        specs=specs,
        execution_specs=execution_specs,
        output_dir=output_dir,
        ollama_base_url=ollama_base_url,
    )
    missing_local = [
        spec.id
        for spec in execution_specs
        if spec.provider == "ollama" and spec.id not in inventory
    ]
    if missing_local:
        raise RuntimeError("Ollama models missing before run start: " + ", ".join(missing_local))

    remote_inventory = (
        openrouter_inventory(openrouter_base_url)
        if any(spec.provider == "openrouter" for spec in execution_specs)
        else set()
    )
    missing_remote = [
        spec.id
        for spec in execution_specs
        if spec.provider == "openrouter" and spec.id not in remote_inventory
    ]
    if missing_remote:
        raise RuntimeError(
            "OpenRouter models missing before run start: " + ", ".join(missing_remote)
        )

    selected_by_dataset = {
        dataset: select_examples(
            load_dataset(dataset, dataset_root),
            limit=limit,
            example_ids=(example_ids_by_dataset or {}).get(dataset),
        )
        for dataset in datasets
    }
    frozen_plan = _frozen_plan(
        phase=phase,
        specs=specs,
        datasets=datasets,
        pipelines=pipelines,
        shot_conditions=shot_conditions,
        repetitions=repetitions,
        repetition_start=repetition_start,
        dataset_root=dataset_root,
        selected_by_dataset=selected_by_dataset,
        settings=settings,
        inventory=inventory,
    )
    planned = (
        len(specs)
        * sum(len(items) for items in selected_by_dataset.values())
        * len(pipelines)
        * len(shot_conditions)
        * repetitions
    )
    created_at = datetime.now(timezone.utc)
    manifest = {
        "schema_version": 1,
        "run_id": created_at.strftime("%Y%m%dT%H%M%S%fZ"),
        "phase": phase,
        "created_at_utc": created_at.isoformat(),
        "plan_fingerprint": _json_sha256(frozen_plan),
        "planned_observations": planned,
        "expected_observations_per_cell": {
            dataset: len(items) * len(shot_conditions) * repetitions
            for dataset, items in selected_by_dataset.items()
        },
        "model_order": [spec.id for spec in specs],
        "dataset_order": datasets,
        "pipeline_order": pipelines,
        "shot_condition_order": shot_conditions,
        "environment": _environment_snapshot(),
        "frozen_plan": frozen_plan,
        "credential_environment": {
            "openrouter": "OPENROUTER_API_KEY",
            "gurobi": ["WLSACCESSID", "WLSSECRET", "LICENSEID"],
        },
    }
    store = ResultStore(output_dir, manifest)
    _write_environment_summary(output_dir, store.manifest)
    progress = ProgressDisplay(
        phase=phase,
        model_order=[spec.id for spec in specs],
        dataset_order=datasets,
        pipeline_order=pipelines,
        shot_condition_order=shot_conditions,
        selected_counts={dataset: len(items) for dataset, items in selected_by_dataset.items()},
        repetitions=repetitions,
    )
    executor = GurobiExecutor(settings.code_timeout_seconds)
    if any(pipeline != "direct" for pipeline in pipelines):
        _solver_preflight(executor)

    current: dict[str, Any] | None = None
    store.write_progress(status="running")
    progress.start(store.completed)
    try:
        for spec in execution_specs:
            client = _client_for(
                spec,
                settings=settings,
                ollama_base_url=ollama_base_url,
                openrouter_base_url=openrouter_base_url,
                openrouter_api_key=openrouter_api_key,
            )
            try:
                for dataset in datasets:
                    for repetition in range(
                        repetition_start, repetition_start + repetitions
                    ):
                        for shot_condition in shot_conditions:
                            for index, example in enumerate(selected_by_dataset[dataset]):
                                for pipeline in _balanced_order(pipelines, index, repetition):
                                    key = (
                                        phase,
                                        spec.id,
                                        dataset,
                                        example.example_id,
                                        pipeline,
                                        repetition,
                                        shot_condition,
                                    )
                                    if key in store.completed:
                                        continue
                                    current = {
                                        "model": spec.id,
                                        "dataset": dataset,
                                        "example_id": example.example_id,
                                        "pipeline": pipeline,
                                        "shot_condition": shot_condition,
                                        "repetition": repetition,
                                    }
                                    store.write_progress(status="running", current=current)
                                    progress.update(store.completed, current=current)
                                    record = run_pipeline(
                                        run_id=store.run_id,
                                        phase=phase,
                                        repetition=repetition,
                                        shot_condition=shot_condition,
                                        pipeline=pipeline,
                                        example=example,
                                        client=client,
                                        executor=executor,
                                        absolute_tolerance=settings.absolute_tolerance,
                                        relative_tolerance=settings.relative_tolerance,
                                    )
                                    if _is_insufficient_credit_record(record):
                                        record.terminal_outcome = "insufficient_credits"
                                        if record.stages:
                                            record.stages[-1]["error_type"] = (
                                                "insufficient_credits"
                                            )
                                        store.append_infrastructure_diagnostic(record)
                                        message = (
                                            "OpenRouter has insufficient credits for "
                                            f"{spec.id}. No observation checkpoint was written. "
                                            "Add credits to the account, then retry this same "
                                            "observation."
                                        )
                                        store.write_progress(
                                            status="waiting_for_credits",
                                            current=current,
                                            interruption=message,
                                        )
                                        progress.finish(
                                            store.completed,
                                            status="waiting_for_credits",
                                            current=current,
                                        )
                                        raise BenchmarkWaitingForCredits(message)
                                    if record.terminal_outcome in INFRASTRUCTURE_TERMINALS:
                                        store.append_infrastructure_diagnostic(record)
                                        raise RuntimeError(
                                            "Infrastructure outcome "
                                            f"{record.terminal_outcome} for {spec.id}; "
                                            "the observation remains incomplete and will be retried."
                                        )
                                    store.append(record)
                                    store.write_progress(status="running")
                                    progress.update(store.completed, current=current)
                                    current = None
                                    progress.update(store.completed)
            finally:
                client.close()
    except KeyboardInterrupt:
        store.write_progress(
            status="interrupted",
            current=current,
            interruption=(
                "User interrupt received. The current unfinished observation was not "
                "checkpointed and will be retried on the next execution."
            ),
        )
        progress.finish(store.completed, status="interrupted", current=current)
        raise BenchmarkInterrupted(
            "Benchmark interrupted after durable checkpoint finalization."
        ) from None
    except BenchmarkWaitingForCredits:
        raise
    except Exception as error:
        store.write_progress(
            status="failed",
            current=current,
            interruption=f"Infrastructure failure: {type(error).__name__}: {error}",
        )
        progress.finish(store.completed, status="failed", current=current)
        raise

    status = "complete" if len(store.completed) == planned else "selection_complete"
    store.write_progress(status=status)
    progress.finish(store.completed, status=status)
    return store.ledger_path, store.manifest_path


def describe_plan(
    specs: list[ModelSpec],
    datasets: list[str],
    pipelines: list[PipelineName],
    shot_conditions: list[ShotCondition],
    dataset_root: Path,
    limit: int | None,
    repetitions: int,
    example_ids_by_dataset: dict[str, set[str]] | None = None,
    repetition_start: int = 1,
    execution_provider: str = "all",
) -> dict[str, Any]:
    counts = {
        dataset: len(
            select_examples(
                load_dataset(dataset, dataset_root),
                limit=limit,
                example_ids=(example_ids_by_dataset or {}).get(dataset),
            )
        )
        for dataset in datasets
    }
    observations = (
        len(specs)
        * sum(counts.values())
        * len(pipelines)
        * len(shot_conditions)
        * repetitions
    )
    execution_specs = select_provider_models(specs, execution_provider)
    scheduled_observations = (
        len(execution_specs)
        * sum(counts.values())
        * len(pipelines)
        * len(shot_conditions)
        * repetitions
    )
    return {
        "model_order": [spec.id for spec in specs],
        "dataset_order": datasets,
        "models": [asdict(spec) for spec in specs],
        "dataset_counts": counts,
        "pipelines": pipelines,
        "shot_conditions": shot_conditions,
        "repetitions": repetitions,
        "repetition_start": repetition_start,
        "pipeline_item_observations": observations,
        "execution_provider": execution_provider,
        "execution_model_order": [spec.id for spec in execution_specs],
        "scheduled_observations": scheduled_observations,
    }


def _ollama_inventory_for_plan(
    *,
    specs: list[ModelSpec],
    execution_specs: list[ModelSpec],
    output_dir: Path,
    ollama_base_url: str,
) -> dict[str, dict[str, Any]]:
    if not any(spec.provider == "ollama" for spec in specs):
        return {}
    if any(spec.provider == "ollama" for spec in execution_specs):
        return ollama_inventory(ollama_base_url)

    manifest_path = output_dir / "_state" / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        models = manifest.get("frozen_plan", {}).get("models", [])
        return {
            str(model["id"]): model["ollama_metadata"]
            for model in models
            if isinstance(model, dict)
            and model.get("provider") == "ollama"
            and isinstance(model.get("ollama_metadata"), dict)
        }

    # A new full-plan result directory still needs the local model metadata in
    # its frozen plan. Subsequent API-only resumes reuse that stored metadata.
    return ollama_inventory(ollama_base_url)


def _is_insufficient_credit_record(record: Any) -> bool:
    if record.provider != "openrouter":
        return False
    for stage in record.stages:
        if stage.get("status_code") == 402:
            return True
        details = stage.get("details")
        excerpt = details.get("response_excerpt", "") if isinstance(details, dict) else ""
        normalized = str(excerpt).casefold()
        if "insufficient credit" in normalized or "credit balance" in normalized:
            return True
    return False


def _client_for(
    spec: ModelSpec,
    *,
    settings: ExperimentSettings,
    ollama_base_url: str,
    openrouter_base_url: str,
    openrouter_api_key: str,
):
    common = {
        "timeout_seconds": settings.request_timeout_seconds,
        "retry_timeouts": settings.retry_request_timeouts,
        "max_tokens": settings.max_output_tokens,
        "temperature": settings.temperature,
        "seed": settings.seed,
        "max_retries": settings.max_transport_retries,
    }
    if spec.provider == "ollama":
        return OllamaClient(spec.id, base_url=ollama_base_url, **common)
    return OpenRouterClient(
        spec.id,
        openrouter_api_key,
        base_url=openrouter_base_url,
        **common,
    )


def _balanced_order(
    pipelines: list[PipelineName], index: int, repetition: int
) -> list[PipelineName]:
    if not pipelines:
        return []
    offset = (index + repetition - 1) % len(pipelines)
    return pipelines[offset:] + pipelines[:offset]


def _frozen_plan(
    *,
    phase: str,
    specs: list[ModelSpec],
    datasets: list[str],
    pipelines: list[PipelineName],
    shot_conditions: list[ShotCondition],
    repetitions: int,
    repetition_start: int,
    dataset_root: Path,
    selected_by_dataset: dict[str, list[Any]],
    settings: ExperimentSettings,
    inventory: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_directory = Path(__file__).resolve().parent
    return {
        "phase": phase,
        "settings": asdict(settings),
        "models": [
            {
                **asdict(spec),
                "ollama_metadata": inventory.get(spec.id)
                if spec.provider == "ollama"
                else None,
            }
            for spec in specs
        ],
        "datasets": [
            {
                "name": dataset,
                "selected_count": len(selected_by_dataset[dataset]),
                "selected_example_ids": [
                    example.example_id for example in selected_by_dataset[dataset]
                ],
                "source_hashes": _dataset_hashes(dataset, dataset_root),
            }
            for dataset in datasets
        ],
        "pipelines": pipelines,
        "shot_conditions": shot_conditions,
        "repetitions": repetitions,
        "repetition_start": repetition_start,
        "runtime_source_hashes": {
            name: file_sha256(source_directory / name)
            for name in (
                "prompts.py",
                "pipelines.py",
                "clients.py",
                "executor.py",
                "datasets.py",
                "runner.py",
                "artifacts.py",
                "records.py",
                "extraction.py",
                "gurobi_runtime.py",
            )
        },
    }


def _dataset_hashes(dataset: str, root: Path) -> dict[str, str]:
    names = {
        "ComplexOR": ["ComplexOR.json"],
        "IndustryOR": ["IndustryOR.json"],
        "LogiOR": ["LogiOR.json"],
    }[dataset]
    return {name: file_sha256(root / name) for name in names}


def _solver_preflight(executor: GurobiExecutor) -> None:
    code = """import gurobipy as gp
from gurobipy import GRB
model = gp.Model(env=env)
x = model.addVar(lb=0.0, name='x')
model.setObjective(x, GRB.MINIMIZE)
model.optimize()
if model.Status == GRB.OPTIMAL:
    print(f'FINAL_OBJECTIVE={model.ObjVal:.17g}')
"""
    result = executor.execute(code)
    if not result.success or result.objective_value != 0.0:
        raise RuntimeError(
            f"Gurobi preflight failed ({result.outcome}): {result.stderr[-500:]}"
        )


def _environment_snapshot() -> dict[str, Any]:
    try:
        import gurobipy as gp

        gurobi_version: Any = list(gp.gurobi.version())
    except (ImportError, AttributeError):
        gurobi_version = None
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "ollama_version": _command_output(["ollama", "--version"]),
        "nvidia_gpu": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "gurobi_version": gurobi_version,
    }


def _write_environment_summary(output_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = manifest["environment"]
    settings = manifest["frozen_plan"]["settings"]
    lines = [
        "# Execution environment",
        "",
        f"- Captured at: {environment.get('captured_at_utc')}",
        f"- Platform: {environment.get('platform')}",
        f"- Processor: {environment.get('processor')}",
        f"- GPU: {environment.get('nvidia_gpu')}",
        f"- Ollama: {environment.get('ollama_version')}",
        f"- Gurobi: {environment.get('gurobi_version')}",
        f"- Model request timeout: {settings['request_timeout_seconds']} seconds",
        f"- Retry request timeouts: {settings['retry_request_timeouts']}",
        f"- Generated-code timeout: {settings['code_timeout_seconds']} seconds",
        f"- Transport retries: {settings['max_transport_retries']}",
        "",
        "## Ollama models",
        "",
        "| Model | Parameters | Quantization | Digest |",
        "|---|---|---|---|",
    ]
    for model in manifest["frozen_plan"]["models"]:
        metadata = model.get("ollama_metadata")
        if not metadata:
            continue
        details = metadata.get("details") or {}
        lines.append(
            f"| {model['id']} | {details.get('parameter_size')} | "
            f"{details.get('quantization_level')} | `{metadata.get('digest')}` |"
        )
    (output_dir / "ENVIRONMENT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    (output_dir / "ENVIRONMENT.json").write_text(
        json.dumps(
            {
                "environment": environment,
                "settings": settings,
                "models": manifest["frozen_plan"]["models"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output if completed.returncode == 0 and output else None
