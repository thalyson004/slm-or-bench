"""Command-line interface for dry runs, smoke tests, and frozen full runs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

from .datasets import SUPPORTED_DATASETS
from .runner import (
    BenchmarkInterrupted,
    BenchmarkWaitingForCredits,
    describe_plan,
    load_model_specs,
    load_settings,
    run_benchmark,
    select_models,
)
from .prompts import SHOT_CONDITIONS

PIPELINES = ("direct", "code", "formulation_code")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument(
        "--provider",
        choices=("all", "ollama", "openrouter"),
        default="all",
        help="Run all selected models or only the selected provider without changing the frozen plan.",
    )
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--pipelines", nargs="+", choices=PIPELINES, default=list(PIPELINES))
    parser.add_argument(
        "--shot-conditions", nargs="+", choices=SHOT_CONDITIONS,
        default=list(SHOT_CONDITIONS),
        help="Prompt conditions evaluated for every model, dataset, and pipeline.",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--repetition-start", type=int, default=1)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int)
    selection.add_argument("--full", action="store_true")
    selection.add_argument("--example-ids", nargs="+")
    selection.add_argument("--panel", choices=("s0", "stability", "formulation_audit"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    load_dotenv(repo / ".env")
    dataset_root = args.dataset_root or repo / "datasets"
    output_dir = args.output_dir or repo / "results"
    settings = load_settings(repo / "configs" / "experiment.json")
    specs = select_models(load_model_specs(repo / "configs" / "models.json"), args.models)
    if args.example_ids and len(args.datasets) != 1:
        raise SystemExit("--example-ids requires exactly one dataset; use --panel for multi-dataset selections.")
    panel_ids = None
    if args.panel:
        panel_payload = json.loads((repo / "configs" / "panels.json").read_text(encoding="utf-8"))
        panel_ids = {
            dataset: set(panel_payload[args.panel][dataset]) for dataset in args.datasets
        }
    elif args.example_ids:
        panel_ids = {args.datasets[0]: set(args.example_ids)}
    limit = None if args.full or panel_ids else (args.limit or 1)
    phase = (
        "S0" if args.panel == "s0"
        else "R1" if args.panel == "stability"
        else "AUDIT" if args.panel == "formulation_audit"
        else "P1" if args.full
        else "DEV"
    )
    plan = describe_plan(
        specs,
        args.datasets,
        args.pipelines,
        args.shot_conditions,
        dataset_root,
        limit,
        args.repetitions,
        panel_ids,
        args.repetition_start,
        args.provider,
    )
    plan["phase"] = phase
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    print(
        f"Starting {phase} benchmark: {plan['pipeline_item_observations']} observations "
        f"in the frozen plan; {plan['scheduled_observations']} assigned to this "
        f"{args.provider} execution ({len(plan['execution_model_order'])} models)."
    )
    print(f"Output: {output_dir}")
    _install_interrupt_handlers()
    while True:
        try:
            jsonl, manifest = run_benchmark(
                phase=phase,
                specs=specs,
                datasets=args.datasets,
                pipelines=args.pipelines,
                shot_conditions=args.shot_conditions,
                repetitions=args.repetitions,
                repetition_start=args.repetition_start,
                dataset_root=dataset_root,
                output_dir=output_dir,
                settings=settings,
                limit=limit,
                example_ids_by_dataset=panel_ids,
                ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
                execution_provider=args.provider,
            )
            break
        except BenchmarkInterrupted as error:
            print(str(error))
            print(f"Resume by running the same command with output directory: {output_dir}")
            return 130
        except BenchmarkWaitingForCredits as error:
            print(str(error))
            if not sys.stdin.isatty():
                print(f"Resume after adding credits with output directory: {output_dir}")
                return 75
            try:
                input("Add OpenRouter credits, then press Enter to retry the unfinished observation. ")
            except (EOFError, KeyboardInterrupt):
                print(f"\nResume after adding credits with output directory: {output_dir}")
                return 130
    print(f"Results: {jsonl}")
    print(f"Manifest: {manifest}")
    return 0


def _install_interrupt_handlers() -> None:
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGBREAK", "SIGTSTP"):
        candidate = getattr(signal, name, None)
        if candidate is not None:
            signal.signal(candidate, interrupt)


if __name__ == "__main__":
    raise SystemExit(main())
