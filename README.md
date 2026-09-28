# Benchmarking Small Language Models on Operations Research

This repository contains the code, benchmark inputs, and response-level artifacts for a comparison of three locally served language models on operations-research problems. The released study covers only `gpt-oss:20b`, `deepseek-r1:14b`, and `qwen3:14b`.

## Study scope

Each of the 192 instances was attempted once in each of three pipelines and two prompt conditions, producing 3,456 observations across 54 complete model–dataset–pipeline–condition cells.

| Model | Provider | Quantization recorded at run time |
| --- | --- | --- |
| `gpt-oss:20b` | Ollama | MXFP4 |
| `deepseek-r1:14b` | Ollama | Q4_K_M |
| `qwen3:14b` | Ollama | Q4_K_M |

| Dataset | Instances |
| --- | ---: |
| ComplexOR | 18 |
| LogiOR | 92 |
| IndustryOR | 82 |
| **Total** | **192** |

| Pipeline | Requested model output | Evaluation |
| --- | --- | --- |
| P1 (`direct`) | Objective value | Compare the returned value with the dataset target |
| P2 (`code`) | Gurobi program | Check and execute the program; parse and score its objective |
| P3 (`formulation_code`) | Mathematical formulation, then Gurobi program | Pass the formulation to the code-generation stage, then check and execute the program |

Both `zero_shot` and `few_shot` conditions are included. Few-shot prompts use the same two solved demonstrations, which are embedded in the versioned prompt source and are not evaluation instances. The complete prompt text is in `src/or_small_models/prompts.py`.

## Response and execution records

The selected observation set is stored as three ordered shards in `artifacts/journal/records-001.jsonl` through `records-003.jsonl`. Each JSONL row is keyed by model, dataset, instance, pipeline, prompt condition, phase, and repetition. It retains the rendered prompts and returned response data when available, provider metadata, intermediate formulations and code, execution output, objective checks, terminal outcome, latency, and available token counts. Fields depend on whether a stage was reached and what the serving interface returned; missing values are not imputed. Concatenating the shards in numeric order reconstructs the journal byte-for-byte and matches the digest in the release manifest.

The same observations are grouped for inspection by dataset, pipeline, and model:

```text
artifacts/
|-- release_manifest.json
|-- ENVIRONMENT.json
|-- journal/records-{001..003}.jsonl
|-- ComplexOR/p1/gpt-oss_20b/
|   |-- artifacts.json
|   `-- summary.md
`-- ...
```

`artifacts/ENVIRONMENT.json` records the run environment and model digests and quantization metadata. The provider-returned `thinking` field, when present, is preserved as text. It is not a verified account of internal model cognition. `artifacts/release_manifest.json` records the exact release scope and a SHA-256 digest of the selected journal.

These records allow independent checks of reported outcomes, regeneration of descriptive summaries, comparisons across pipelines or prompt conditions, and alternative error-label analyses without repeating inference. Training a diagnostic classifier or using examples in a later pipeline study are possible future applications, not experiments reported here.

## Repository contents

| Path | Contents |
| --- | --- |
| `src/or_small_models/` | Clients, prompts, pipelines, code execution, checkpointing, and result summaries |
| `configs/` | Three-model order, pipeline settings, and benchmark panels |
| `datasets/` | ComplexOR, LogiOR, and IndustryOR inputs |
| `analysis/` | Validation, result summarization, table and figure generation, and code reevaluation |
| `tests/` | Deterministic tests for the runner, prompts, artifacts, and provider response parsing |
| `artifacts/` | Released three-model observations, per-cell artifacts, summaries, and environment metadata |
| `run.ps1` | PowerShell entry point for planning and benchmark execution |

The included results are the three-model article subset. New runs are written to `results/` (full runs) or `results-dev/` (development runs); these execution directories are separate from the released records in `artifacts/`.

## Requirements and setup

- Python 3.11 or newer;
- Ollama and the three model tags in the table above;
- Gurobi and a valid local license for P2 and P3 execution.

```powershell
python -m pip install -e ".[solver,analysis]"
python -m unittest discover -s tests -v
```

The runner uses `http://localhost:11434` by default. It does not require an API key for this three-model release. Configure a different Ollama endpoint with `OLLAMA_BASE_URL` if needed. Do not commit local credentials or `.env` files.

Download the exact model tags before a full run:

```powershell
ollama pull gpt-oss:20b
ollama pull deepseek-r1:14b
ollama pull qwen3:14b
```

## Run and resume

Preview a one-instance-per-dataset plan:

```powershell
.\run.ps1
```

Run the full configured matrix:

```powershell
.\run.ps1 -Full -Execute
```

The progress display reads completed observations from the checkpoint journal. Interrupt the process with `Ctrl+C`; run the same command and output directory to resume. A new run is written separately from the immutable released artifacts.

## Validate and analyze the released records

```powershell
python analysis/validate_release.py
python analysis/summarize_results.py artifacts/journal --output-dir analysis/generated
python analysis/validate_primary.py --summary analysis/generated/summary.csv
python analysis/generate_tables.py --summary analysis/generated/summary.csv --output analysis/generated/accuracy_table.csv
python analysis/generate_figures.py --summary analysis/generated/summary.csv --output analysis/generated/accuracy.png
```

The summary command creates `analysis/generated/summary.csv`; pass that file to `validate_primary.py`. Generated files are not part of the observation source of truth. Individual execution records and their grouped views are in `artifacts/`.

## Citation

Pending.

## Validity

Pending.

## License

Pending.
