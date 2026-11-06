"""Train LSTM-Autoencoder for all window sizes, seeds, and architectures."""

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
    """Run a command with streaming stdout/stderr and return exit code, stdout, stderr."""
    print_dim(f"Running: {' '.join(cmd)}")
    print()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True,
    )

    stdout_lines = []
    stderr_lines = []

    # Stream stdout in real-time
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line, end='', flush=True)
                stdout_lines.append(line)

    # Stream stderr in real-time
    if process.stderr:
        for line in iter(process.stderr.readline, ''):
            if line:
                print(line, end='', flush=True, file=sys.stderr)
                stderr_lines.append(line)

    process.wait()
    return_code = process.returncode

    stdout = ''.join(stdout_lines)
    stderr = ''.join(stderr_lines)

    return return_code, stdout, stderr


def main() -> int:
    """Train all configurations."""
    project_root = Path.cwd()
    dataset = "paysim_synthetic_financial"
    windows = [1, 3, 5, 10]
    seeds = [42, 123, 2026]
    architectures = ["attention", "bottleneck", "plain", "attention_pooling"]

    seed_list = ",".join(map(str, seeds))

    print_header("Starting comprehensive training for all configurations")
    print()
    print_info(f"Dataset: {dataset}")
    print_info(f"Windows: {windows}")
    print_info(f"Seeds: {seeds}")
    print_info(f"Architectures: {architectures}")
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

        # Check if checkpoints already exist
        if checkpoint_exists(project_root, window, arch_name, seeds):
            print_header(f"[{current}/{total_configs}] Skipping: Window={window}, Architecture={arch_name}, Seeds={seed_list}")  # noqa: E501
            print()
            print_warning("⊘ Checkpoints already exist, skipping...")
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
            "--seeds", seed_list,
        ] + arch_flags

        try:
            return_code, stdout, stderr = run_command(cmd)
            print()
            print_success(f"✓ Completed: Window={window}, Architecture={arch_name}")
            print()
            completed += 1
        except Exception as e:
            print()
            print_error(f"✗ Error: Command failed with exception: {e}")
            print_error(f"Command: {' '.join(cmd)}")
            print()
            failed += 1
            continue

        # Check return code manually since we're not using check=True
        if return_code != 0:
            print()
            print_error(f"✗ Error: Command failed with exit code {return_code}")
            print_error(f"Command: {' '.join(cmd)}")

            if stdout:
                print_error("STDOUT:")
                print_error(stdout)

            if stderr:
                print_error("STDERR:")
                print_error(stderr)

            print()
            failed += 1
            continue

    print_header("Training Summary")
    print()
    print_info(f"Total configurations: {total_configs}")
    print_success(f"Completed: {completed}")
    print_warning(f"Skipped (already exists): {skipped}")
    if failed > 0:
        print_error(f"Failed: {failed}")
    print()

    if failed == 0:
        print_success("All training runs completed successfully!")
    else:
        print_warning(f"Training completed with {failed} failures.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
