from __future__ import annotations

import json
import importlib.util
import csv
from io import StringIO
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from or_small_models.clients import OllamaClient, OpenRouterClient
from or_small_models.artifacts import ResultStore, safe_component
from or_small_models.datasets import load_dataset, load_json_values, select_examples
from or_small_models.errors import ModelRequestError
from or_small_models.executor import (
    CodeSafetyError,
    GurobiExecutor,
    _minimal_environment,
    validate_generated_code,
)
from or_small_models.extraction import (
    extract_direct_objective,
    extract_python_code,
    objectives_match,
)
from or_small_models.pipelines import run_pipeline
from or_small_models.progress import ProgressDisplay
from or_small_models.prompts import (
    SYSTEM,
    appendix_prompt_documents,
    code_prompt,
    direct_prompt,
    formulation_code_prompt,
    formulation_prompt,
)
from or_small_models.runner import (
    BenchmarkInterrupted,
    BenchmarkWaitingForCredits,
    ExperimentSettings,
    describe_plan,
    load_model_specs,
    run_benchmark,
    select_models,
    select_provider_models,
)
from or_small_models.records import BenchmarkRecord, DatasetExample, ModelResponse, ModelSpec


class DatasetTests(unittest.TestCase):
    def test_loads_json_array_and_concatenated_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            array = root / "array.json"
            stream = root / "stream.json"
            array.write_text('[{"id": 1}, {"id": 2}]', encoding="utf-8")
            stream.write_text('{"id": 1}\n{"id": 2}', encoding="utf-8")
            self.assertEqual([item["id"] for item in load_json_values(array)], [1, 2])
            self.assertEqual([item["id"] for item in load_json_values(stream)], [1, 2])

    def test_selection_is_bounded_by_default_policy(self):
        examples = [DatasetExample("D", str(i), "q", float(i)) for i in range(3)]
        self.assertEqual([item.example_id for item in select_examples(examples, limit=1)], ["0"])
        self.assertEqual(
            [item.example_id for item in select_examples(examples, example_ids={"2"})],
            ["2"],
        )

    def test_limit_and_ids_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            select_examples(
                [DatasetExample("D", "x", "q", 1.0)],
                limit=1,
                example_ids={"x"},
            )

    def test_repository_dataset_denominators(self):
        root = Path(__file__).resolve().parents[1] / "datasets"
        self.assertEqual(len(load_dataset("ComplexOR", root)), 18)
        self.assertEqual(len(load_dataset("LogiOR", root)), 92)
        self.assertEqual(len(load_dataset("IndustryOR", root)), 82)
        for name in ("ComplexOR", "LogiOR", "IndustryOR"):
            self.assertTrue((root / f"{name}.json").is_file())
        self.assertFalse((root / "ComplexOR_ground_truth.json").exists())
        self.assertFalse((root / "IndustryOR_ground_truth.json").exists())
        logi_first = load_json_values(root / "LogiOR.json")[0]
        self.assertIn("expected_objective", logi_first)


class ExtractionTests(unittest.TestCase):
    def test_direct_output_requires_one_finite_marker(self):
        self.assertEqual(extract_direct_objective("FINAL_OBJECTIVE=1.25e2"), 125.0)
        self.assertIsNone(extract_direct_objective("answer: 125"))
        self.assertIsNone(
            extract_direct_objective("FINAL_OBJECTIVE=1\nFINAL_OBJECTIVE=2")
        )

    def test_code_extraction_accepts_one_fence(self):
        self.assertEqual(extract_python_code("```python\nprint(1)\n```"), "print(1)")
        with self.assertRaises(ValueError):
            extract_python_code("```python\na=1\n```\n```python\nb=2\n```")

    def test_relative_or_absolute_tolerance(self):
        self.assertTrue(objectives_match(100.5, 100.0, absolute_tolerance=0.01, relative_tolerance=0.01))
        self.assertFalse(objectives_match(102.0, 100.0, absolute_tolerance=0.01, relative_tolerance=0.01))


class PromptTests(unittest.TestCase):
    question = "Maximize x subject to 0 <= x <= 1."

    def test_zero_and_few_shot_prompts_are_distinct(self):
        self.assertIn("ZERO-SHOT CONDITION", direct_prompt(self.question, "zero_shot"))
        self.assertIn("FEW-SHOT DEMONSTRATION", direct_prompt(self.question, "few_shot"))
        self.assertIn("FEW-SHOT DEMONSTRATION", code_prompt(self.question, "few_shot"))
        self.assertIn("FEW-SHOT DEMONSTRATION", formulation_prompt(self.question, "few_shot"))
        self.assertIn(
            "FEW-SHOT DEMONSTRATION",
            formulation_code_prompt(self.question, "Sets and indices: I = {1}.", "few_shot"),
        )

    def test_unknown_shot_condition_is_rejected(self):
        with self.assertRaises(ValueError):
            direct_prompt(self.question, "invalid")

    def test_few_shot_prompts_share_complete_worked_examples(self):
        prompts = (
            direct_prompt(self.question, "few_shot"),
            code_prompt(self.question, "few_shot"),
            formulation_prompt(self.question, "few_shot"),
            formulation_code_prompt(
                self.question, "Sets and indices: I = {1}.", "few_shot"
            ),
        )
        for prompt in prompts:
            self.assertIn("minimum total allocation cost", prompt)
            self.assertIn("minimum total investment cost", prompt)
        self.assertIn("FINAL_OBJECTIVE=10000", prompts[0])
        self.assertIn("FINAL_OBJECTIVE=20", prompts[0])
        self.assertIn("model.addConstr(x - y >= 200", prompts[1])
        self.assertIn("model.addConstr(4 * solar + wind <= 15", prompts[1])
        self.assertIn("minimize Z = 50 x_X + 30 x_Y", prompts[2])
        self.assertIn("minimize Z = 7 x_S + 5 x_W", prompts[2])
        self.assertIn("model.addConstr(x + y <= 1000", prompts[3])
        self.assertIn("model.addConstr(2 * solar + 3 * wind >= 10", prompts[3])

    def test_release_prompt_inventory_is_rendered_from_runtime_templates(self):
        documents = appendix_prompt_documents()
        self.assertEqual(documents["system-prompt.txt"], SYSTEM)
        self.assertEqual(
            documents["p1-zero-shot.txt"], direct_prompt("{{question}}", "zero_shot")
        )
        self.assertEqual(
            documents["p3-code-few-shot.txt"],
            formulation_code_prompt("{{question}}", "{{formulation}}", "few_shot"),
        )


class ExecutorPolicyTests(unittest.TestCase):
    def test_minimal_environment_preserves_user_site_packages(self):
        environment = _minimal_environment()
        import site

        self.assertIn(site.getusersitepackages(), environment.get("PYTHONPATH", ""))

    def test_accepts_default_or_injected_gurobi_environment(self):
        validate_generated_code("import gurobipy as gp\nm = gp.Model()")
        validate_generated_code("import gurobipy as gp\nm = gp.Model(env=env)")

    def test_allows_main_guard_and_throwaway_dunder_loop_variable(self):
        validate_generated_code(
            "import gurobipy as gp\n"
            "m = gp.Model(env=env)\n"
            "for __ in range(1):\n    pass\n"
            "if __name__ == '__main__':\n    m.optimize()\n"
        )

    def test_blocks_file_and_process_access(self):
        for code in (
            "import os\nimport gurobipy as gp\nm=gp.Model(env=env)",
            "import gurobipy as gp\nm=gp.Model(env=env)\nopen('x','w')",
        ):
            with self.assertRaises(CodeSafetyError):
                validate_generated_code(code)
        with self.assertRaises(CodeSafetyError):
            validate_generated_code(
                "import gurobipy as gp\nm=gp.Model(env=env)\n__import__('os')"
            )

    def test_safety_rejection_has_specific_outcome(self):
        result = GurobiExecutor().execute("import subprocess")
        self.assertEqual(result.outcome, "blocked_import")
        self.assertFalse(result.success)

    def test_code_without_gurobi_constructor_and_unused_sys_import_is_allowed(self):
        validate_generated_code("import sys\nprint('FINAL_OBJECTIVE=1')")

    def test_sys_attribute_access_has_specific_outcome(self):
        result = GurobiExecutor().execute("import sys\nprint(sys.modules)")
        self.assertEqual(result.outcome, "blocked_module_access")
        self.assertFalse(result.success)


class ClientTests(unittest.TestCase):
    def test_ollama_timeout_is_typed_and_not_retried(self):
        def timeout(_request):
            raise httpx.ReadTimeout("slow generation")

        client = OllamaClient(
            "qwen3:14b",
            timeout_seconds=600,
            max_retries=2,
            client=httpx.Client(
                transport=httpx.MockTransport(timeout), base_url="http://test/"
            ),
            clock=iter([1.0, 3.5]).__next__,
        )
        with self.assertRaises(ModelRequestError) as captured:
            client.generate("system", "prompt")
        self.assertEqual(captured.exception.code, "request_timeout")
        self.assertEqual(captured.exception.attempts, 1)
        self.assertEqual(captured.exception.latency_seconds, 2.5)

    def test_ollama_response_metadata(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "message": {"content": "FINAL_OBJECTIVE=3"},
                    "prompt_eval_count": 10,
                    "eval_count": 2,
                    "done_reason": "stop",
                },
            )
        )
        client = OllamaClient(
            "qwen3:14b",
            client=httpx.Client(transport=transport, base_url="http://test/"),
            clock=iter([1.0, 2.5]).__next__,
        )
        response = client.generate("system", "prompt")
        self.assertEqual(response.prompt_tokens, 10)
        self.assertEqual(response.completion_tokens, 2)
        self.assertEqual(response.latency_seconds, 1.5)

    def test_openrouter_response_metadata(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "id": "response-1",
                    "model": "test/remote-model",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                },
            )
        )
        client = OpenRouterClient(
            "test/remote-model",
            "secret-for-test",
            client=httpx.Client(transport=transport, base_url="http://test/"),
            clock=iter([1.0, 2.0]).__next__,
        )
        response = client.generate("system", "prompt")
        self.assertEqual(response.model, "test/remote-model")
        self.assertEqual(response.prompt_tokens, 5)
        self.assertNotIn("secret-for-test", json.dumps(response.provider_metadata))

    def test_openrouter_payment_required_preserves_http_402(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                402, json={"error": {"message": "Insufficient credits"}}
            )
        )
        client = OpenRouterClient(
            "test/remote-model",
            "secret-for-test",
            client=httpx.Client(transport=transport, base_url="http://test/"),
            clock=iter([1.0, 2.0]).__next__,
        )
        with self.assertRaises(ModelRequestError) as captured:
            client.generate("system", "prompt")
        self.assertEqual(captured.exception.code, "provider_http_error")
        self.assertEqual(captured.exception.status_code, 402)
        self.assertEqual(captured.exception.attempts, 1)


class StubClient:
    model = "stub:1b"
    provider = "ollama"

    def __init__(self, outputs: list[str], finish_reason: str | None = None):
        self.outputs = iter(outputs)
        self.finish_reason = finish_reason

    def generate(self, system: str, prompt: str) -> ModelResponse:
        return ModelResponse(
            content=next(self.outputs),
            model=self.model,
            provider=self.provider,
            latency_seconds=0.1,
            prompt_tokens=5,
            completion_tokens=2,
            finish_reason=self.finish_reason,
        )


class StubExecutor:
    def execute(self, code: str):
        from or_small_models.records import ExecutionResult

        return ExecutionResult("executed", True, objective_value=42.0, latency_seconds=0.2)


class PipelineTests(unittest.TestCase):
    example = DatasetExample("Test", "1", "Find the optimum.", 42.0)

    def run_one(self, name, outputs):
        return run_pipeline(
            run_id="run",
            phase="TEST",
            repetition=1,
            pipeline=name,
            example=self.example,
            client=StubClient(outputs),
            executor=StubExecutor(),
            absolute_tolerance=0.01,
            relative_tolerance=0.01,
        )

    def test_direct_pipeline(self):
        record = self.run_one("direct", ["FINAL_OBJECTIVE=42"])
        self.assertTrue(record.correct)
        self.assertEqual(record.shot_condition, "zero_shot")
        self.assertEqual(len(record.stages), 1)

    def test_code_pipeline(self):
        record = self.run_one("code", ["import gurobipy as gp\nm=gp.Model(env=env)"])
        self.assertTrue(record.correct)
        self.assertEqual([stage["stage"] for stage in record.stages], ["code_generation", "execution"])

    def test_formulation_code_pipeline_uses_exactly_two_calls(self):
        record = self.run_one(
            "formulation_code",
            ["Variables, objective, constraints", "import gurobipy as gp\nm=gp.Model(env=env)"],
        )
        self.assertTrue(record.correct)
        self.assertEqual(len([stage for stage in record.stages if "response" in stage]), 2)

    def test_invalid_direct_contract_is_not_corrected(self):
        record = self.run_one("direct", ["The answer is 42."])
        self.assertEqual(record.terminal_outcome, "parse_error")
        self.assertFalse(record.correct)

    def test_truncated_code_is_not_executed(self):
        record = run_pipeline(
            run_id="run",
            phase="TEST",
            repetition=1,
            pipeline="code",
            example=self.example,
            client=StubClient(["partial code"], finish_reason="length"),
            executor=StubExecutor(),
            absolute_tolerance=0.01,
            relative_tolerance=0.01,
        )
        self.assertEqual(record.terminal_outcome, "truncated_response")
        self.assertEqual(len(record.stages), 1)

    def test_request_timeout_preserves_typed_diagnostic(self):
        class TimeoutClient(StubClient):
            def generate(self, system: str, prompt: str):
                raise ModelRequestError(
                    code="request_timeout",
                    message="request exceeded 600 seconds",
                    provider="ollama",
                    model=self.model,
                    attempts=1,
                    latency_seconds=600.0,
                )

        record = run_pipeline(
            run_id="run",
            phase="TEST",
            repetition=1,
            pipeline="direct",
            example=self.example,
            client=TimeoutClient([]),
            executor=StubExecutor(),
            absolute_tolerance=0.01,
            relative_tolerance=0.01,
        )
        self.assertEqual(record.terminal_outcome, "request_timeout")
        self.assertEqual(record.stages[0]["error_type"], "request_timeout")
        self.assertEqual(record.stages[0]["user_prompt"], direct_prompt("Find the optimum."))


class PlanTests(unittest.TestCase):
    def test_manifest_contains_all_requested_models(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "models.json"
        specs = load_model_specs(path)
        self.assertEqual(len(specs), 3)
        self.assertEqual(
            [spec.id for spec in specs],
            [
                "gpt-oss:20b", "deepseek-r1:14b", "qwen3:14b",
            ],
        )
        self.assertTrue(all(spec.provider == "ollama" for spec in specs))
        self.assertEqual(
            [spec.id for spec in select_provider_models(specs, "ollama")],
            [spec.id for spec in specs if spec.provider == "ollama"],
        )

    def test_frozen_panel_sizes(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "panels.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(sum(len(ids) for ids in payload["s0"].values()), 3)
        self.assertEqual(sum(len(ids) for ids in payload["stability"].values()), 30)
        self.assertEqual(sum(len(ids) for ids in payload["formulation_audit"].values()), 15)

    def test_plan_includes_both_prompt_conditions(self):
        repository = Path(__file__).resolve().parents[1]
        dataset_root = repository / "datasets"
        specs = select_models(load_model_specs(repository / "configs" / "models.json"), ["qwen3:14b"])
        plan = describe_plan(
            specs,
            ["ComplexOR"],
            ["direct"],
            ["zero_shot", "few_shot"],
            dataset_root,
            limit=1,
            repetitions=1,
        )
        self.assertEqual(plan["shot_conditions"], ["zero_shot", "few_shot"])
        self.assertEqual(plan["dataset_order"], ["ComplexOR"])
        self.assertEqual(plan["pipeline_item_observations"], 2)

    def test_progress_display_reports_resume_and_nested_bars(self):
        stream = StringIO()
        display = ProgressDisplay(
            phase="P1",
            model_order=["m"],
            dataset_order=["ComplexOR"],
            pipeline_order=["direct"],
            shot_condition_order=["zero_shot", "few_shot"],
            selected_counts={"ComplexOR": 2},
            repetitions=1,
            stream=stream,
        )
        display.is_tty = True
        completed = {
            ("P1", "m", "ComplexOR", "example-1", "direct", 1, "zero_shot")
        }
        display.start(completed)
        display.update(
            completed,
            current={
                "model": "m",
                "dataset": "ComplexOR",
                "example_id": "example-2",
                "pipeline": "direct",
                "shot_condition": "few_shot",
            },
        )
        rendered = stream.getvalue()
        self.assertIn("Benchmark P1", rendered)
        self.assertIn("1/4", rendered)
        self.assertIn("model m:", rendered)
        self.assertIn("dataset ComplexOR:", rendered)
        self.assertIn("pipeline direct:", rendered)
        self.assertIn("prompt zero_shot:", rendered)
        self.assertIn("Current  m | ComplexOR | direct | few_shot", rendered)


class ArtifactTests(unittest.TestCase):
    def test_checkpoint_resume_and_release_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "run_id": "run-1",
                "plan_fingerprint": "frozen",
                "planned_observations": 2,
                "expected_observations_per_cell": {"ComplexOR": 2},
            }
            store = ResultStore(root, manifest)
            store.append(
                BenchmarkRecord(
                    run_id="run-1", phase="P1", repetition=1,
                    dataset="ComplexOR", example_id="1", difficulty=None,
                    problem_type=None, model="gpt-oss:20b", provider="ollama",
                    pipeline="code", expected_objective=1.0,
                    predicted_objective=1.0, correct=True,
                    terminal_outcome="correct", stages=[{"response": "code"}],
                    total_latency_seconds=2.0, prompt_tokens=10,
                    completion_tokens=5, shot_condition="zero_shot",
                )
            )
            store.write_progress(status="interrupted")
            cell = root / "ComplexOR" / "p2" / "gpt-oss_20b"
            artifact_before_resume = (cell / "artifacts.json").read_text(
                encoding="utf-8"
            )
            summary_before_resume = (cell / "summary.md").read_text(
                encoding="utf-8"
            )
            resumed = ResultStore(root, {**manifest, "run_id": "ignored"})
            self.assertEqual(len(resumed.completed), 1)
            payload = json.loads((cell / "artifacts.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["model"], "gpt-oss:20b")
            self.assertEqual(payload["records"][0]["stages"][0]["response"], "code")
            self.assertIn("Completed observations: 1 / 2", (cell / "summary.md").read_text(encoding="utf-8"))
            self.assertEqual(safe_component("test/remote-model"), "test_remote-model")
            self.assertEqual(
                artifact_before_resume,
                (cell / "artifacts.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                summary_before_resume,
                (cell / "summary.md").read_text(encoding="utf-8"),
            )

    def test_interrupted_resume_allows_progress_renderer_change_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = {
                "phase": "P1",
                "settings": {"temperature": 0.0},
                "models": [{"id": "m"}],
                "datasets": [{"name": "D", "selected_count": 1}],
                "pipelines": ["direct"],
                "shot_conditions": ["zero_shot"],
                "repetitions": 1,
                "repetition_start": 1,
                "runtime_source_hashes": {
                    "runner.py": "old", "artifacts.py": "old", "executor.py": "old",
                    "prompts.py": "same"
                },
            }
            old_manifest = {
                "run_id": "old",
                "plan_fingerprint": "old-fingerprint",
                "planned_observations": 1,
                "frozen_plan": frozen,
            }
            old_store = ResultStore(root, old_manifest)
            old_store.write_progress(status="interrupted")
            new_frozen = json.loads(json.dumps(frozen))
            new_frozen["runtime_source_hashes"]["runner.py"] = "new"
            new_frozen["runtime_source_hashes"]["artifacts.py"] = "new"
            new_frozen["runtime_source_hashes"]["executor.py"] = "new"
            new_manifest = {
                **old_manifest,
                "run_id": "new",
                "plan_fingerprint": "new-fingerprint",
                "frozen_plan": new_frozen,
            }
            resumed = ResultStore(root, new_manifest)
            self.assertEqual(resumed.run_id, "old")
            self.assertEqual(resumed.manifest["plan_fingerprint"], "new-fingerprint")
            self.assertEqual(
                resumed.manifest["frozen_plan"]["runtime_source_hashes"]["runner.py"],
                "new",
            )
            self.assertEqual(len(resumed.manifest["resume_compatibility_events"]), 1)

    def test_resume_rejects_changed_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_manifest = {
                "run_id": "old",
                "plan_fingerprint": "old-fingerprint",
                "planned_observations": 1,
                "frozen_plan": {"settings": {"temperature": 0.0}},
            }
            old_store = ResultStore(root, old_manifest)
            old_store.write_progress(status="interrupted")
            new_manifest = {
                **old_manifest,
                "plan_fingerprint": "new-fingerprint",
                "frozen_plan": {"settings": {"temperature": 0.5}},
            }
            with self.assertRaises(ValueError):
                ResultStore(root, new_manifest)

    def test_interrupted_run_retries_only_unfinished_observation(self):
        repository = Path(__file__).resolve().parents[1]
        settings = ExperimentSettings(
            temperature=0.0,
            seed=0,
            max_output_tokens=128,
            request_timeout_seconds=600,
            retry_request_timeouts=False,
            code_timeout_seconds=10,
            max_transport_retries=2,
            absolute_tolerance=0.01,
            relative_tolerance=0.01,
        )
        spec = ModelSpec("stub:1b", "ollama", 1.0, "stub", "test")

        class InterruptingClient:
            model = "stub:1b"
            provider = "ollama"

            def generate(self, _system, _prompt):
                raise KeyboardInterrupt

            def close(self):
                return None

        class SuccessfulClient(InterruptingClient):
            def generate(self, _system, _prompt):
                return ModelResponse(
                    content="FINAL_OBJECTIVE=11",
                    model=self.model,
                    provider=self.provider,
                    latency_seconds=0.1,
                )

        inventory = {
            "stub:1b": {
                "digest": "digest",
                "size_bytes": 1,
                "modified_at": "frozen",
                "details": {"parameter_size": "1B", "quantization_level": "Q4"},
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            common = {
                "phase": "P1",
                "specs": [spec],
                "datasets": ["ComplexOR"],
                "pipelines": ["direct"],
                "shot_conditions": ["zero_shot"],
                "repetitions": 1,
                "repetition_start": 1,
                "dataset_root": repository / "datasets",
                "output_dir": output,
                "settings": settings,
                "limit": 1,
                "example_ids_by_dataset": None,
                "ollama_base_url": "http://test",
                "openrouter_base_url": "http://test",
                "openrouter_api_key": "",
            }
            with patch("or_small_models.runner.ollama_inventory", return_value=inventory), patch(
                "or_small_models.runner._environment_snapshot", return_value={}
            ), patch(
                "or_small_models.runner._client_for", return_value=InterruptingClient()
            ):
                with self.assertRaises(BenchmarkInterrupted):
                    run_benchmark(**common)
            self.assertFalse((output / "_state" / "records.jsonl").exists())
            interrupted = json.loads(
                (output / "_state" / "progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(interrupted["status"], "interrupted")

            with patch("or_small_models.runner.ollama_inventory", return_value=inventory), patch(
                "or_small_models.runner._environment_snapshot", return_value={}
            ), patch(
                "or_small_models.runner._client_for", return_value=SuccessfulClient()
            ):
                run_benchmark(**common)
            records = (output / "_state" / "records.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(records), 1)
            complete = json.loads(
                (output / "_state" / "progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(complete["status"], "complete")

    def test_api_credit_failure_is_not_checkpointed_and_retries_same_observation(self):
        repository = Path(__file__).resolve().parents[1]
        first = load_dataset("ComplexOR", repository / "datasets")[0]
        settings = ExperimentSettings(
            temperature=0.0,
            seed=0,
            max_output_tokens=128,
            request_timeout_seconds=600,
            retry_request_timeouts=False,
            code_timeout_seconds=10,
            max_transport_retries=2,
            absolute_tolerance=0.01,
            relative_tolerance=0.01,
        )
        local = ModelSpec("stub:1b", "ollama", 1.0, "stub", "test")
        remote = ModelSpec(
            "test/remote-model", "openrouter", None, "test", "test"
        )

        class CreditClient:
            model = remote.id
            provider = "openrouter"

            def generate(self, _system, _prompt):
                raise ModelRequestError(
                    code="provider_http_error",
                    message="OpenRouter rejected the request (HTTP 402).",
                    provider=self.provider,
                    model=self.model,
                    attempts=1,
                    latency_seconds=0.1,
                    status_code=402,
                    details={"response_excerpt": "Insufficient credits"},
                )

            def close(self):
                return None

        class SuccessfulClient(CreditClient):
            def generate(self, _system, _prompt):
                return ModelResponse(
                    content=f"FINAL_OBJECTIVE={first.expected_objective}",
                    model=self.model,
                    provider=self.provider,
                    latency_seconds=0.1,
                )

        inventory = {
            local.id: {
                "digest": "digest",
                "size_bytes": 1,
                "modified_at": "frozen",
                "details": {"parameter_size": "1B", "quantization_level": "Q4"},
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            common = {
                "phase": "P1",
                "specs": [local, remote],
                "datasets": ["ComplexOR"],
                "pipelines": ["direct"],
                "shot_conditions": ["zero_shot"],
                "repetitions": 1,
                "repetition_start": 1,
                "dataset_root": repository / "datasets",
                "output_dir": output,
                "settings": settings,
                "limit": 1,
                "example_ids_by_dataset": None,
                "ollama_base_url": "http://test",
                "openrouter_base_url": "http://test",
                "openrouter_api_key": "secret-for-test",
                "execution_provider": "openrouter",
            }
            patches = (
                patch("or_small_models.runner.ollama_inventory", return_value=inventory),
                patch(
                    "or_small_models.runner.openrouter_inventory",
                    return_value={remote.id},
                ),
                patch("or_small_models.runner._environment_snapshot", return_value={}),
            )
            with patches[0], patches[1], patches[2], patch(
                "or_small_models.runner._client_for", return_value=CreditClient()
            ):
                with self.assertRaises(BenchmarkWaitingForCredits):
                    run_benchmark(**common)

            self.assertFalse((output / "_state" / "records.jsonl").exists())
            waiting = json.loads(
                (output / "_state" / "progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(waiting["status"], "waiting_for_credits")
            diagnostic = json.loads(
                (output / "_state" / "infrastructure_errors.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(
                diagnostic["observation"]["terminal_outcome"],
                "insufficient_credits",
            )

            with patch(
                "or_small_models.runner.openrouter_inventory", return_value={remote.id}
            ), patch(
                "or_small_models.runner._environment_snapshot", return_value={}
            ), patch(
                "or_small_models.runner._client_for", return_value=SuccessfulClient()
            ), patch(
                "or_small_models.runner.ollama_inventory"
            ) as local_inventory:
                run_benchmark(**common)
            local_inventory.assert_not_called()
            records = (output / "_state" / "records.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(records), 1)
            completed = json.loads(records[0])
            self.assertEqual(completed["model"], remote.id)
            final_progress = json.loads(
                (output / "_state" / "progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(final_progress["status"], "selection_complete")

    def test_ollama_only_execution_does_not_require_or_contact_openrouter(self):
        repository = Path(__file__).resolve().parents[1]
        first = load_dataset("ComplexOR", repository / "datasets")[0]
        settings = ExperimentSettings(
            temperature=0.0,
            seed=0,
            max_output_tokens=128,
            request_timeout_seconds=600,
            retry_request_timeouts=False,
            code_timeout_seconds=10,
            max_transport_retries=2,
            absolute_tolerance=0.01,
            relative_tolerance=0.01,
        )
        local = ModelSpec("stub:1b", "ollama", 1.0, "stub", "test")
        remote = ModelSpec(
            "test/remote-model", "openrouter", None, "test", "test"
        )

        class LocalClient:
            model = local.id
            provider = "ollama"

            def generate(self, _system, _prompt):
                return ModelResponse(
                    content=f"FINAL_OBJECTIVE={first.expected_objective}",
                    model=self.model,
                    provider=self.provider,
                    latency_seconds=0.1,
                )

            def close(self):
                return None

        inventory = {
            local.id: {
                "digest": "digest",
                "size_bytes": 1,
                "modified_at": "frozen",
                "details": {"parameter_size": "1B", "quantization_level": "Q4"},
            }
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "or_small_models.runner.ollama_inventory", return_value=inventory
        ), patch(
            "or_small_models.runner.openrouter_inventory"
        ) as remote_inventory, patch(
            "or_small_models.runner._environment_snapshot", return_value={}
        ), patch(
            "or_small_models.runner._client_for", return_value=LocalClient()
        ):
            output = Path(directory)
            run_benchmark(
                phase="P1",
                specs=[local, remote],
                datasets=["ComplexOR"],
                pipelines=["direct"],
                shot_conditions=["zero_shot"],
                repetitions=1,
                repetition_start=1,
                dataset_root=repository / "datasets",
                output_dir=output,
                settings=settings,
                limit=1,
                example_ids_by_dataset=None,
                ollama_base_url="http://test",
                openrouter_base_url="http://test",
                openrouter_api_key="",
                execution_provider="ollama",
            )
            remote_inventory.assert_not_called()
            records = (output / "_state" / "records.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(records), 1)
            self.assertEqual(json.loads(records[0])["model"], local.id)


class AnalysisTests(unittest.TestCase):
    def test_generated_code_reevaluation_removes_migration_metadata(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "analysis"
            / "reevaluate_generated_code.py"
        )
        spec = importlib.util.spec_from_file_location("reevaluate_generated_code", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class Executor:
            def execute(self, _code):
                from or_small_models.records import ExecutionResult

                return ExecutionResult(
                    outcome="executed",
                    success=True,
                    objective_value=42.0,
                    latency_seconds=0.2,
                )

        record = BenchmarkRecord(
            run_id="r",
            phase="P1",
            repetition=1,
            dataset="D",
            example_id="1",
            difficulty=None,
            problem_type=None,
            model="m",
            provider="ollama",
            pipeline="code",
            expected_objective=42.0,
            predicted_objective=None,
            correct=False,
            terminal_outcome="policy_error",
            stages=[
                {"stage": "code_generation", "response": "code", "latency_seconds": 1.0},
                {
                    "stage": "execution",
                    "code": "import gurobipy as gp\nm = gp.Model()",
                    "outcome": "policy_error",
                    "success": False,
                    "stderr": "Every Gurobi model must be created with env=env.",
                    "latency_seconds": 0.01,
                },
            ],
            total_latency_seconds=1.01,
            prompt_tokens=2,
            completion_tokens=1,
        ).to_dict()
        record["terminal_outcome"] = "correct"
        record["predicted_objective"] = 42.0
        record["correct"] = True
        record["stages"][1]["outcome"] = "executed"
        record["reevaluations"] = [{"type": "policy_error_reexecution"}]
        updated = module.reevaluate_record(
            record,
            executor=Executor(),
            absolute_tolerance=0.01,
            relative_tolerance=0.01,
        )
        self.assertEqual(updated["terminal_outcome"], "correct")
        self.assertEqual(updated["predicted_objective"], 42.0)
        self.assertTrue(updated["correct"])
        self.assertEqual(updated["stages"][0], record["stages"][0])
        self.assertNotIn("reevaluations", updated)

    def test_jsonl_summary_preserves_denominator(self):
        script = Path(__file__).resolve().parents[1] / "analysis" / "summarize_results.py"
        spec = importlib.util.spec_from_file_location("summarize_results", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "records.jsonl"
            records = [
                {
                    "run_id": "r", "phase": "P1", "repetition": 1, "dataset": "D", "example_id": str(i),
                    "difficulty": None, "problem_type": None, "model": "m", "provider": "ollama",
                    "pipeline": "direct", "expected_objective": 1, "predicted_objective": 1 if i == 0 else 2,
                    "correct": i == 0, "terminal_outcome": "correct" if i == 0 else "incorrect_objective",
                    "total_latency_seconds": i + 1, "prompt_tokens": 2, "completion_tokens": 1,
                }
                for i in range(2)
            ]
            source.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
            loaded = module.load_records([source])
            summary = root / "summary.csv"
            module.write_summary(loaded, summary)
            with summary.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["n"], "2")
            self.assertEqual(row["correct"], "1")
            self.assertEqual(float(row["accuracy"]), 0.5)

    def test_primary_validator_rejects_partial_summary(self):
        script = Path(__file__).resolve().parents[1] / "analysis" / "validate_primary.py"
        spec = importlib.util.spec_from_file_location("validate_primary", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with self.assertRaises(ValueError):
            module.validate_primary_rows([], Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    unittest.main()
