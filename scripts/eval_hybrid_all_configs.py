"""Evaluate hybrid classifier for all window sizes, seeds, and architectures."""

import argparse
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

from lstm_autoencoder.reporting.beautify_log import (
    print_dim,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)

DEFAULT_WINDOWS = [1, 3, 5, 10]
DEFAULT_SEEDS = [42, 123, 2026]
ARCHITECTURES = ["attention", "bottleneck", "plain", "attention_pooling"]


def _csv_ints(value: str) -> list[int]:
    """Parse a comma-separated list of integers, e.g. '1,3,5'."""
    return [int(item) for item in value.split(",") if item.strip()]


def _csv_architectures(value: str) -> list[str]:
    """Parse a comma-separated list of architecture names and validate each."""
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in ARCHITECTURES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown architecture(s): {unknown}. Choose from: {ARCHITECTURES}"
        )
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options that select which configurations to evaluate."""
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid classifier for a grid of windows, seeds, and architectures."
    )
    parser.add_argument(
        "--architectures", type=_csv_architectures, default=list(ARCHITECTURES),
        help=f"Comma-separated architectures to evaluate (default: all). Choices: {', '.join(ARCHITECTURES)}",
    )
    parser.add_argument(
        "--windows", type=_csv_ints, default=list(DEFAULT_WINDOWS),
        help="Comma-separated window sizes (default: 1,3,5,10).",
    )
    parser.add_argument(
        "--seeds", type=_csv_ints, default=list(DEFAULT_SEEDS),
        help="Comma-separated seeds (default: 42,123,2026).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-evaluate even if reports already exist (overwrites them).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands that would run (and which would be skipped) without evaluating.",
    )
    parser.add_argument(
        "--nrows", type=int, default=None,
        help="Read only the first N rows (quick trial run).",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=200,
        help="Paired bootstrap resamples per seed for the delta CIs (0 to skip).",
    )
    return parser.parse_args(argv)


def get_report_path(
    project_root: Path,
    window: int,
    arch_name: str,
    seeds: list[int],
) -> Path:
    """Get the report path for a specific configuration."""
    arch_tag = {
        "attention": "attn",
        "bottleneck": "bottleneck",
        "plain": "plain",
        "attention_pooling": "attn_pool",
    }[arch_name]
    # Report path structure: artifacts/report/paysim_synthetic_financial/hybrid_classifier/<timestamp>/window=*_architecture=*/hybrid_classifier_report.json
    # We check if any report exists for this (window, architecture) combination
    report_dir = project_root / "artifacts" / "report" / "paysim_synthetic_financial" / "hybrid_classifier"
    return report_dir


def get_architecture_flags(arch_name: str) -> list[str]:
    """Get CLI flags for an architecture name."""
    if arch_name == "attention":
        return []
    elif arch_name == "bottleneck":
        return ["--bottleneck"]
    elif arch_name == "plain":
        return ["--plain"]
    elif arch_name == "attention_pooling":
        return ["--attention-pooling"]
    else:
        return []


def report_exists(
    project_root: Path,
    window: int,
    arch_name: str,
) -> bool:
    """Check if a report for a configuration exists."""
    report_dir = get_report_path(project_root, window, arch_name, [])
    if not report_dir.exists():
        return False
    # Check if any timestamp folder contains a report with matching window and architecture
    for timestamp_dir in report_dir.iterdir():
        if not timestamp_dir.is_dir():
            continue
        for config_dir in timestamp_dir.iterdir():
            if not config_dir.is_dir():
                continue
            # Check if config_dir name matches window and architecture
            expected_pattern = f"window={window}_architecture={arch_name}"
            if config_dir.name == expected_pattern:
                json_path = config_dir / "hybrid_classifier_report.json"
                if json_path.exists():
                    return True
    return False


def run_command(cmd: list[str]) -> tuple[int, str, str]:
    """Run a command, stream its combined output live, and return (exit code, output, "")."""
    print_dim(f"Running: {' '.join(cmd)}")
    print()

    # Set environment for UTF-8 encoding on Windows
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True,
        env=env,
        encoding="utf-8",
        errors="replace",  # Replace encoding errors instead of raising
    )

    output_lines: list[str] = []
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            print(line, end="", flush=True)
            output_lines.append(line)

    process.wait()
    return process.returncode, "".join(output_lines), ""


def main(argv: list[str] | None = None) -> int:
    """Evaluate the selected configurations."""
    args = parse_args(argv)
    project_root = Path.cwd()
    dataset = "paysim_synthetic_financial"
    windows: list[int] = args.windows
    seeds: list[int] = args.seeds
    architectures: list[str] = args.architectures

    seed_list = ",".join(map(str, seeds))

    print_header("Starting hybrid classifier evaluation for selected configurations")
    print()
    print_info(f"Dataset: {dataset}")
    print_info(f"Windows: {windows}")
    print_info(f"Seeds: {seeds}")
    print_info(f"Architectures: {architectures}")
    print_info(f"Force re-evaluate: {args.force}")
    print_info(f"Bootstrap samples: {args.bootstrap}")
    if args.nrows:
        print_info(f"Row limit: {args.nrows}")
    if args.dry_run:
        print_warning("Dry run: nothing will be evaluated")
    print()
    print_dim(f"Total configurations: {len(windows) * len(architectures)}")
    print()
    print_info(f"Project root: {project_root}")
    print()

    total_configs = len(windows) * len(architectures)
    current = 0
    skipped = 0
    completed = 0
    failed = 0

    for window, arch_name in product(windows, architectures):
        current += 1
        arch_flags = get_architecture_flags(arch_name)

        # Resume mechanism: skip configurations whose reports already exist (unless --force)
        if not args.force and report_exists(project_root, window, arch_name):
            print_header(f"[{current}/{total_configs}] Skipping: Window={window}, Architecture={arch_name}")
            print()
            print_warning("- Report already exists, skipping (use --force to re-evaluate)...")
            print()
            skipped += 1
            continue

        print_header(f"[{current}/{total_configs}] Evaluating: Window={window}, Architecture={arch_name}, Seeds={seed_list}")
        print()

        cmd = [
            "uv", "run", "python", "-m", "lstm_autoencoder", "eval-hybrid",
            "--dataset", dataset,
            "--window", str(window),
        ] + arch_flags + ["--seeds", seed_list, "--bootstrap", str(args.bootstrap)]

        if args.nrows:
            cmd.extend(["--nrows", str(args.nrows)])

        if args.dry_run:
            print_dim(f"Would run: {' '.join(cmd)}")
            print()
            continue

        try:
            return_code, output, _ = run_command(cmd)
        except Exception as e:
            print()
            print_error(f"X Error: Command failed with exception: {e}")
            print_error(f"Command: {' '.join(cmd)}")
            print()
            failed += 1
            continue

        if return_code != 0:
            print()
            print_error(f"X Error: Command failed with exit code {return_code}")
            print_error(f"Command: {' '.join(cmd)}")
            print()
            failed += 1
            continue

        print()
        print_success(f"OK Completed: Window={window}, Architecture={arch_name}")
        print()
        completed += 1

    print_header("Evaluation Summary")
    print()
    print_info(f"Total configurations: {total_configs}")
    print_success(f"Completed: {completed}")
    print_warning(f"Skipped (already exists): {skipped}")
    if failed > 0:
        print_error(f"Failed: {failed}")
    print()

    if failed == 0:
        print_success("Selected evaluation runs completed successfully!")
    else:
        print_warning(f"Evaluation completed with {failed} failures.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
