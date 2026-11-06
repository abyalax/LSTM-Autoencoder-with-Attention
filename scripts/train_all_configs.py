"""Train LSTM-Autoencoder for all window sizes, seeds, and architectures."""

import argparse
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
    """Parse CLI options that select which configurations to train."""
    parser = argparse.ArgumentParser(
        description="Train LSTM-Autoencoder for a grid of windows, seeds, and architectures."
    )
    parser.add_argument(
        "--architectures", type=_csv_architectures, default=list(ARCHITECTURES),
        help=f"Comma-separated architectures to train (default: all). Choices: {', '.join(ARCHITECTURES)}",
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
        help="Retrain even if checkpoints already exist (overwrites them). "
             "Needed when existing checkpoints of a selected architecture are invalid.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands that would run (and which would be skipped) without training.",
    )
    return parser.parse_args(argv)


def get_checkpoint_path(
    project_root: Path,
    window: int,
    arch_name: str,
    seed: int,
) -> Path:
    """Get the checkpoint path for a specific configuration."""
    arch_tag = {
        "attention": "attn",
        "bottleneck": "bottleneck",
        "plain": "plain",
        "attention_pooling": "attn_pool",
    }[arch_name]
    checkpoint_name = f"lstm_ae_w{window}_{arch_tag}_seed{seed}_best.pt"
    return project_root / "artifacts" / "model" / "paysim_synthetic_financial" / checkpoint_name


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


def checkpoint_exists(
    project_root: Path,
    window: int,
    arch_name: str,
    seeds: list[int],
) -> bool:
    """Check if all checkpoints for a configuration exist."""
    for seed in seeds:
        checkpoint_path = get_checkpoint_path(project_root, window, arch_name, seed)
        if not checkpoint_path.exists():
            return False
    return True


def run_command(cmd: list[str]) -> tuple[int, str, str]:
    """Run a command, stream its combined output live, and return (exit code, output, "").

    stderr is merged into stdout: reading the two pipes one after another can
    deadlock once the unread pipe's OS buffer fills up during a long training run.
    """
    print_dim(f"Running: {' '.join(cmd)}")
    print()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True,
    )

    output_lines: list[str] = []
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            print(line, end="", flush=True)
            output_lines.append(line)

    process.wait()
    return process.returncode, "".join(output_lines), ""


def main(argv: list[str] | None = None) -> int:
    """Train the selected configurations."""
    args = parse_args(argv)
    project_root = Path.cwd()
    dataset = "paysim_synthetic_financial"
    windows: list[int] = args.windows
    seeds: list[int] = args.seeds
    architectures: list[str] = args.architectures

    seed_list = ",".join(map(str, seeds))

    print_header("Starting training for selected configurations")
    print()
    print_info(f"Dataset: {dataset}")
    print_info(f"Windows: {windows}")
    print_info(f"Seeds: {seeds}")
    print_info(f"Architectures: {architectures}")
    print_info(f"Force retrain: {args.force}")
    if args.dry_run:
        print_warning("Dry run: nothing will be trained")
    print()
    print_dim(f"Total configurations: {len(windows) * len(architectures)} (x {len(seeds)} seeds each)")
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

        # Resume mechanism: skip configurations whose checkpoints all exist (unless --force)
        if not args.force and checkpoint_exists(project_root, window, arch_name, seeds):
            print_header(f"[{current}/{total_configs}] Skipping: Window={window}, Architecture={arch_name}, Seeds={seed_list}")  # noqa: E501
            print()
            print_warning("⊘ Checkpoints already exist, skipping (use --force to retrain)...")
            for seed in seeds:
                checkpoint_path = get_checkpoint_path(project_root, window, arch_name, seed)
                print_dim(f"  - {checkpoint_path.name}")
            print()
            skipped += 1
            continue

        print_header(f"[{current}/{total_configs}] Training: Window={window}, Architecture={arch_name}, Seeds={seed_list}")  # noqa: E501
        print()

        cmd = [
            "uv", "run", "python", "-m", "lstm_autoencoder", "train-model",
            "--dataset", dataset,
            "--window", str(window),
        ] + arch_flags + ["--seeds", seed_list]

        if args.dry_run:
            print_dim(f"Would run: {' '.join(cmd)}")
            print()
            continue

        try:
            return_code, output, _ = run_command(cmd)
        except Exception as e:
            print()
            print_error(f"✗ Error: Command failed with exception: {e}")
            print_error(f"Command: {' '.join(cmd)}")
            print()
            failed += 1
            continue

        if return_code != 0:
            print()
            print_error(f"✗ Error: Command failed with exit code {return_code}")
            print_error(f"Command: {' '.join(cmd)}")
            print()
            failed += 1
            continue

        print()
        print_success(f"✓ Completed: Window={window}, Architecture={arch_name}")
        print()
        completed += 1

    print_header("Training Summary")
    print()
    print_info(f"Total configurations: {total_configs}")
    print_success(f"Completed: {completed}")
    print_warning(f"Skipped (already exists): {skipped}")
    if failed > 0:
        print_error(f"Failed: {failed}")
    print()

    if failed == 0:
        print_success("Selected training runs completed successfully!")
    else:
        print_warning(f"Training completed with {failed} failures.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
