# Released observations

`journal/records-001.jsonl` through `journal/records-003.jsonl` are the ordered, size-limited shards of the observation source of truth. Together they contain 3,456 unique records for `gpt-oss:20b`, `deepseek-r1:14b`, and `qwen3:14b`, with 1,152 records per model. All 54 model–dataset–pipeline–prompt-condition cells are complete. Concatenate the shards in numeric order to reconstruct the original journal; the combined SHA-256 is recorded in `release_manifest.json`.

The fields `phase`, `model`, `dataset`, `example_id`, `pipeline`, `repetition`, and `shot_condition` form the experimental key. Each row stores the expected and predicted objective, terminal outcome, correctness label, latency and available token counts. The `stages` array preserves stage-specific prompts, returned response text and raw provider payloads when available, provider metadata, mathematical formulations, generated programs, and deterministic execution records. A stage may be absent when it was not reached; provider fields may be absent when no response or metadata was returned.

## Record fields

At the observation level, each JSON object contains:

| Field | Meaning |
| --- | --- |
| `run_id`, `phase`, `repetition` | Source run and execution identifiers |
| `model`, `provider` | Requested model tag and serving provider |
| `dataset`, `example_id`, `difficulty`, `problem_type` | Benchmark instance identifiers and available descriptors |
| `pipeline`, `shot_condition` | Requested pathway and prompt condition |
| `expected_objective`, `predicted_objective`, `correct` | Reference target, parsed prediction when available, and numeric agreement under the configured tolerance |
| `terminal_outcome` | Final parser, model, execution, or scoring status |
| `stages` | Ordered model-call and deterministic execution records |
| `total_latency_seconds`, `prompt_tokens`, `completion_tokens` | Observation-level timing and available token totals |

Model-call stage records preserve `system_prompt`, `user_prompt`, normalized `response`, provider-reported `thinking` and other `provider_metadata`, the raw provider response, resolved model identifier, request attempts, latency, token counts, and finish reason when returned. Execution-stage records preserve generated `code`, `outcome`, `success`, `objective_value`, return code, standard output and standard error, and execution latency. The concrete stage fields vary with the pipeline and terminal point; consult `src/or_small_models/records.py` and `src/or_small_models/pipelines.py` for the serialization contract.

`ENVIRONMENT.json` describes the recorded software, hardware, solver, settings, model digests, and quantization. The GPU identifier has been omitted; device model and driver are retained. `release_manifest.json` records the selected run, dimensions, record key, per-model environment metadata, and SHA-256 digest of the released JSONL.

For convenience, each observation is also included in `artifacts/{dataset}/{p1|p2|p3}/{model_directory}/artifacts.json`; `summary.md` reports cell-level descriptive outcomes. These grouped files are views of the same journal, not additional observations.
