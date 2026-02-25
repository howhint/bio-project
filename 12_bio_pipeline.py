from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


PROJECT_DIR = Path(__file__).resolve().parents[1]
CUDA_DIR = PROJECT_DIR / "cuda"
PY_DIR = PROJECT_DIR / "py"


@dataclass(frozen=True)
class Step:
    key: str
    script_path: Path
    description: str
    optional: bool = False
    legacy: bool = False


STEPS: List[Step] = [
    Step(
        key="extract_extinct_animals",
        script_path=PY_DIR / "01_get_extinct_animal.py",
        description="Build extinct animal list from GBIF and write parsed CSVs.",
    ),
    Step(
        key="get_diet_info",
        script_path=PY_DIR / "02_get_diet_info.py",
        description="Fetch predator/prey diet enrichment data.",
    ),
    Step(
        key="missing_data_extracter",
        script_path=PY_DIR / "04_missing_data_extracter.py",
        description="Extract rows that still need manual follow-up.",
    ),
    Step(
        key="backup_pred_prey_finder",
        script_path=PY_DIR / "03_test_back_up_pred_prey_finder.py",
        description="Lookup unknown predator/prey values via Wikidata backup.",
        optional=True,
        legacy=True,
    ),
    Step(
        key="year_conversion",
        script_path=PY_DIR / "05_year_conversion.py",
        description="Build year multipliers and year-pair equivalence table.",
    ),
    Step(
        key="analyze_occurrences",
        script_path=PY_DIR / "06_analyze_occurrences.py",
        description="Compute occurrence counts around extinction years.",
    ),
    Step(
        key="filter_no_zero_animals",
        script_path=PY_DIR / "07_filter_no_zero_animals.py",
        description="Filter raw occurrence rows that contain any zero side counts.",
        optional=True,
    ),
    Step(
        key="compiler",
        script_path=PY_DIR / "08_compiler.py",
        description="Score animal impact percentages.",
    ),
    Step(
        key="results",
        script_path=PY_DIR / "09_results.py",
        description="Aggregate impact results by taxonomic order and stats.",
    ),
    Step(
        key="export_groups_to_google_sheets",
        script_path=PY_DIR / "10_export_groups_to_google_sheets.py",
        description="Upload group outputs to Google Sheets and create charts.",
        optional=True,
    ),
    Step(
        key="top3_unknown_export",
        script_path=PY_DIR / "11_top3_unknown_export.py",
        description="Legacy top-3 export over deprecated summary output.",
        optional=True,
        legacy=True,
    ),
]

STEP_BY_KEY: Dict[str, Step] = {step.key: step for step in STEPS}

DEFAULT_PIPELINE: List[str] = [
    "extract_extinct_animals",
    "get_diet_info",
    "missing_data_extracter",
    "year_conversion",
    "analyze_occurrences",
    "compiler",
    "results",
]


def run_step(step: Step, python_exec: str, dry_run: bool = False) -> int:
    if not step.script_path.exists():
        raise FileNotFoundError(f"Missing script for step '{step.key}': {step.script_path}")

    cmd = [python_exec, str(step.script_path)]
    print(f"\n=== [{step.key}] {step.description}")
    print(f"Command: {' '.join(cmd)}")
    if dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=str(step.script_path.parent))
    return int(completed.returncode)


def resolve_pipeline(
    include_optional: bool,
    include_legacy: bool,
    include_google_export: bool,
    start_at: str,
    end_at: str,
    skip: List[str],
) -> List[Step]:
    ordered_keys = list(DEFAULT_PIPELINE)
    if include_optional:
        ordered_keys.insert(3, "backup_pred_prey_finder")
        ordered_keys.insert(6, "filter_no_zero_animals")
    if include_google_export:
        ordered_keys.append("export_groups_to_google_sheets")
    if include_legacy:
        ordered_keys.append("top3_unknown_export")

    for key in skip:
        if key in ordered_keys:
            ordered_keys = [k for k in ordered_keys if k != key]

    if start_at:
        if start_at not in ordered_keys:
            raise ValueError(f"'--start-at' step not in selected pipeline: {start_at}")
        start_idx = ordered_keys.index(start_at)
        ordered_keys = ordered_keys[start_idx:]
    if end_at:
        if end_at not in ordered_keys:
            raise ValueError(f"'--end-at' step not in selected pipeline: {end_at}")
        end_idx = ordered_keys.index(end_at)
        ordered_keys = ordered_keys[: end_idx + 1]

    if not ordered_keys:
        raise ValueError("Resolved pipeline is empty after applying filters.")

    return [STEP_BY_KEY[key] for key in ordered_keys]


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to run each step (default: current interpreter).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified runner for the school-bio Python pipeline."
    )
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List available steps.")
    list_parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include legacy steps in listing.",
    )

    run_parser = subparsers.add_parser("run", help="Run a single step.")
    run_parser.add_argument("step", choices=sorted(STEP_BY_KEY.keys()))
    add_shared_arguments(run_parser)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run the end-to-end pipeline.")
    pipeline_parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional steps like backup lookup and zero-row filter.",
    )
    pipeline_parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include legacy steps marked from early iterations.",
    )
    pipeline_parser.add_argument(
        "--include-google-export",
        action="store_true",
        help="Run Google Sheets export at the end.",
    )
    pipeline_parser.add_argument(
        "--start-at",
        choices=sorted(STEP_BY_KEY.keys()),
        default="",
        help="Start pipeline from this step key.",
    )
    pipeline_parser.add_argument(
        "--end-at",
        choices=sorted(STEP_BY_KEY.keys()),
        default="",
        help="Stop pipeline after this step key.",
    )
    pipeline_parser.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=sorted(STEP_BY_KEY.keys()),
        help="Skip one step key (repeat for multiple skips).",
    )
    add_shared_arguments(pipeline_parser)

    add_shared_arguments(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    command = args.command or "pipeline"
    python_exec = args.python
    dry_run = bool(args.dry_run)

    if command == "list":
        for step in STEPS:
            if step.legacy and not args.include_legacy:
                continue
            tags = []
            if step.optional:
                tags.append("optional")
            if step.legacy:
                tags.append("legacy")
            suffix = f" ({', '.join(tags)})" if tags else ""
            print(f"- {step.key}: {step.description}{suffix}")
        return

    if command == "run":
        step = STEP_BY_KEY[args.step]
        code = run_step(step, python_exec=python_exec, dry_run=dry_run)
        if code != 0:
            raise SystemExit(code)
        return

    if command == "pipeline":
        selected_steps = resolve_pipeline(
            include_optional=bool(getattr(args, "include_optional", False)),
            include_legacy=bool(getattr(args, "include_legacy", False)),
            include_google_export=bool(getattr(args, "include_google_export", False)),
            start_at=str(getattr(args, "start_at", "") or ""),
            end_at=str(getattr(args, "end_at", "") or ""),
            skip=[str(s) for s in (getattr(args, "skip", []) or [])],
        )
    else:
        selected_steps = [STEP_BY_KEY[key] for key in DEFAULT_PIPELINE]

    print("Resolved pipeline steps:")
    for idx, step in enumerate(selected_steps, start=1):
        print(f"{idx}. {step.key}")

    for step in selected_steps:
        code = run_step(step, python_exec=python_exec, dry_run=dry_run)
        if code != 0:
            print(f"Step failed: {step.key} (exit code {code})")
            raise SystemExit(code)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
