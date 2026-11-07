"""Command-line entry points for the LSTM Autoencoder research project."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.config import ExploratoryConfig
from lstm_autoencoder.data.financial_fraud_detection.exploratory import FinancialFraudDetectionExplorer
from lstm_autoencoder.data.paysim_synthetic_financial.exploratory import PaySimSyntheticFinancialDatasetsExplorer
from lstm_autoencoder.evaluation.config import ProbeConfig, SplitConfig
from lstm_autoencoder.evaluation.financial_fraud_detection.signal_probe import SignalProbe
from lstm_autoencoder.evaluation.paysim_synthetic_financial.baseline import BaselineEvaluator
from lstm_autoencoder.evaluation.paysim_synthetic_financial.hybrid_classifier import (
    HybridClassifierConfig,
    HybridClassifierEvaluator,
)
from lstm_autoencoder.evaluation.paysim_synthetic_financial.hybrid_signal_probe import HybridSignalProbe
from lstm_autoencoder.evaluation.paysim_synthetic_financial.results_summary import (
    DEFAULT_LEGACY_SINCE,
    ResultsSummarizer,
    ResultsSummaryConfig,
)
from lstm_autoencoder.evaluation.paysim_synthetic_financial.signal_probe import (
    PaySimSignalProbe,
)
from lstm_autoencoder.evaluation.paysim_synthetic_financial.unsupervised_baselines import (
    UnsupervisedBaselineConfig,
    UnsupervisedBaselineEvaluator,
)
from lstm_autoencoder.model.config import DenseAutoencoderConfig, ModelConfig, SequenceConfig, TrainConfig
from lstm_autoencoder.model.pipeline import PaySimModelTrainingPipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="lstm-autoencoder",
        description="Research tools for LSTM Autoencoder.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    inspect_parser = subparsers.add_parser(
        "inspect-data",
        help="Explore and summarize the input dataset.",
    )
    _add_dataset_arguments(inspect_parser)

    probe_parser = subparsers.add_parser(
        "probe-signal",
        help="Check whether the fraud label can be learned from the features.",
    )
    ablation_parser = subparsers.add_parser(
        "ablation",
        help="Train full/no-history/history-only baselines and compare per history bucket.",
    )
    _add_dataset_arguments(ablation_parser)
    ablation_parser.add_argument(
        "--no-account-code",
        action="store_true",
        help="Exclude account_code from features (it's a factorized ID, not behavioral).",
    )

    train_parser = subparsers.add_parser(
        "train-model",
        help="Train the LSTM-Autoencoder on PaySim with configurable architecture.",
    )
    _add_dataset_arguments(train_parser)
    train_parser.add_argument("--window", type=int, default=SequenceConfig().window_length)
    train_parser.add_argument(
        "--bottleneck",
        action="store_true",
        help=(
            "Use a single-vector bottleneck decoder (final encoder hidden state, "
            "repeated across the window). Mutually exclusive with --plain and --attention-pooling."
        ),
    )
    train_parser.add_argument(
        "--plain",
        action="store_true",
        help=(
            "Use raw per-step encoder outputs (can copy directly). "
            "Mutually exclusive with --bottleneck and --attention-pooling. Default is attention."
        ),
    )
    train_parser.add_argument(
        "--attention-pooling",
        action="store_true",
        help=(
            "Use attention to pool encoder outputs to a single bottleneck vector, "
            "then repeat across the window. Mutually exclusive with --bottleneck and --plain."
        ),
    )
    train_parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "Comma-separated seeds to repeat training with (e.g. 42,123,2026). "
            "When set, trains once per seed on the same data split and reports "
            "mean/std instead of a single run."
        ),
    )
    hybrid_parser = subparsers.add_parser(
        "probe-hybrid-signal",
        help=(
            "Correlate a trained LSTM-AE checkpoint's reconstruction error with "
            "the LR baseline's score on the same test rows, before retraining a hybrid model."
        ),
    )
    _add_dataset_arguments(hybrid_parser)
    hybrid_parser.add_argument("--window", type=int, default=SequenceConfig().window_length)
    hybrid_parser.add_argument(
        "--bottleneck",
        action="store_true",
        help=(
            "Match the checkpoint's architecture: bottleneck decoder."
            "Mutually exclusive with --plain and --attention-pooling."
        ),
    )
    hybrid_parser.add_argument(
        "--plain",
        action="store_true",
        help=(
            "Match the checkpoint's architecture: raw per-step outputs."
            "Mutually exclusive with --bottleneck and --attention-pooling."
        ),
    )
    hybrid_parser.add_argument(
        "--attention-pooling",
        action="store_true",
        help=(
            "Match the checkpoint's architecture: attention pooling."
            "Mutually exclusive with --bottleneck and --plain."
        ),
    )
    hybrid_parser.add_argument(
        "--seed", type=int, default=42, help="Seed used when the checkpoint was trained (for its filename)."
    )
    hybrid_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the trained checkpoint. Defaults to the standard train-model naming convention.",
    )

    eval_hybrid_parser = subparsers.add_parser(
        "eval-hybrid",
        help=(
            "P3.5: fit LR + LSTM-AE reconstruction error with a leak-free protocol "
            "(second stage fitted on val, judged on test) and compare against the LR baseline."
        ),
    )
    _add_dataset_arguments(eval_hybrid_parser)
    eval_hybrid_parser.add_argument("--window", type=int, default=SequenceConfig().window_length)
    eval_hybrid_parser.add_argument(
        "--bottleneck",
        action="store_true",
        help=(
            "Match the checkpoints' architecture: bottleneck decoder."
            "Mutually exclusive with --plain and --attention-pooling."
        ),
    )
    eval_hybrid_parser.add_argument(
        "--plain",
        action="store_true",
        help=(
            "Match the checkpoints' architecture: raw per-step outputs."
            "Mutually exclusive with --bottleneck and --attention-pooling."
        ),
    )
    eval_hybrid_parser.add_argument(
        "--attention-pooling",
        action="store_true",
        help=(
            "Match the checkpoints' architecture: attention pooling."
            "Mutually exclusive with --bottleneck and --plain."
        ),
    )
    eval_hybrid_parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated seeds of already-trained checkpoints (e.g. 42,123,2026).",
    )
    eval_hybrid_parser.add_argument(
        "--bootstrap",
        type=int,
        default=HybridClassifierConfig.n_bootstrap,
        help="Paired bootstrap resamples per seed for the delta CIs (0 to skip).",
    )

    eval_unsupervised_parser = subparsers.add_parser(
        "eval-unsupervised",
        help=(
            "RQ D1 baselines: Isolation Forest (all features and the 5 sequence features) and a "
            "dense autoencoder on the same split, scored on the same test rows."
        ),
    )
    _add_dataset_arguments(eval_unsupervised_parser)
    eval_unsupervised_parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,2026",
        help="Comma-separated seeds (Isolation Forest random_state and dense AE init/shuffle).",
    )
    eval_unsupervised_parser.add_argument(
        "--keep-account-code",
        action="store_true",
        help=(
            "Keep `account_code` (a factorize id of the receiver) in the Isolation Forest features."
            "Dropped by default."
        ),
    )
    eval_unsupervised_parser.add_argument(
        "--latent-dim",
        type=int,
        default=DenseAutoencoderConfig.latent_dim,
        help="Dense AE latent size; keep it below 5 (the number of input features).",
    )
    eval_unsupervised_parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DenseAutoencoderConfig.hidden_dim,
        help="Dense AE hidden layer size.",
    )

    summarize_parser = subparsers.add_parser(
        "summarize-results",
        help=(
            "Turn the existing JSON reports (training grid, baselines, hybrid classifier) into "
            "markdown tables for the Results section. Reads reports only; nothing is trained."
        ),
    )
    summarize_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing artifacts/.",
    )
    summarize_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset folder name under artifacts/report (e.g., paysim_synthetic_financial).",
    )
    summarize_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output markdown path. Defaults to artifacts/summary/results_summary.md.",
    )
    summarize_parser.add_argument(
        "--legacy-since",
        type=str,
        default=DEFAULT_LEGACY_SINCE,
        help=(
            "Only list unverified training reports (no run_config block) with this "
            "timestamp or later, e.g. 20261106_103302."
        ),
    )

    train_all_parser = subparsers.add_parser(
        "train-all-configs",
        help="Train LSTM-AE for a grid of windows, seeds, and architectures.",
    )
    _add_dataset_arguments(train_all_parser)
    train_all_parser.add_argument(
        "--windows",
        type=str,
        default="1,3,5,10",
        help="Comma-separated window sizes (default: 1,3,5,10).",
    )
    train_all_parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,2026",
        help="Comma-separated seeds (default: 42,123,2026).",
    )
    train_all_parser.add_argument(
        "--architectures",
        type=str,
        default="attention,bottleneck,plain,attention_pooling",
        help="Comma-separated architectures (default: all). Choices: attention, bottleneck, plain, attention_pooling.",
    )
    train_all_parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if checkpoints already exist (overwrites them).",
    )
    train_all_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without training.",
    )

    eval_hybrid_all_parser = subparsers.add_parser(
        "eval-hybrid-all-configs",
        help="Evaluate hybrid classifier for a grid of windows, seeds, and architectures.",
    )
    _add_dataset_arguments(eval_hybrid_all_parser)
    eval_hybrid_all_parser.add_argument(
        "--windows",
        type=str,
        default="1,3,5,10",
        help="Comma-separated window sizes (default: 1,3,5,10).",
    )
    eval_hybrid_all_parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,2026",
        help="Comma-separated seeds (default: 42,123,2026).",
    )
    eval_hybrid_all_parser.add_argument(
        "--architectures",
        type=str,
        default="attention,bottleneck,plain,attention_pooling",
        help="Comma-separated architectures (default: all). Choices: attention, bottleneck, plain, attention_pooling.",
    )
    eval_hybrid_all_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate even if reports already exist (overwrites them).",
    )
    eval_hybrid_all_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without evaluating.",
    )
    eval_hybrid_all_parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Paired bootstrap resamples per seed for the delta CIs (0 to skip).",
    )

    run_pipeline_parser = subparsers.add_parser(
        "run-pipeline",
        help="Run end-to-end pipeline: hybrid evaluation, unsupervised evaluation, summarize.",
    )
    _add_dataset_arguments(run_pipeline_parser)
    run_pipeline_parser.add_argument(
        "--windows",
        type=str,
        default="1,3,5,10",
        help="Comma-separated window sizes (default: 1,3,5,10).",
    )
    run_pipeline_parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,2026",
        help="Comma-separated seeds (default: 42,123,2026).",
    )
    run_pipeline_parser.add_argument(
        "--architectures",
        type=str,
        default="attention,bottleneck,plain,attention_pooling",
        help="Comma-separated architectures (default: all). Choices: attention, bottleneck, plain, attention_pooling.",
    )
    run_pipeline_parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Paired bootstrap resamples per seed for the delta CIs (0 to skip).",
    )

    _add_dataset_arguments(probe_parser)
    probe_parser.add_argument(
        "--max-train-rows",
        type=int,
        default=ProbeConfig.max_train_rows,
        help="Maximum number of training rows used by the probe model.",
    )
    return parser


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing data/raw/.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset folder name in data/raw/ (e.g., financial_fraud_detection).",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Read only the first N rows (quick trial run).",
    )


def _run_inspect_data(
    project_root: Path, dataset: str, nrows: int | None
) -> int:
    """Execute dataset exploration."""
    dataset_config = DatasetConfig(
        project_root=project_root,
        dataset_name=dataset,
        row_limit=nrows,
    )
    exploratory_config = ExploratoryConfig()
    split_config = SplitConfig()
    explorers = {
        "financial_fraud_detection": FinancialFraudDetectionExplorer,
        "paysim_synthetic_financial": PaySimSyntheticFinancialDatasetsExplorer,
    }
    explorer = explorers.get(dataset.lower())
    if explorer is None:
        return 0
    explorer(
        dataset_config=dataset_config,
        exploratory_config=exploratory_config,
        split_config=split_config,
    ).run()
    return 0


def _run_probe_signal(
    project_root: Path, dataset: str, nrows: int | None, max_train_rows: int
) -> int:
    """Execute the signal learnability probe."""
    dataset_config = DatasetConfig(
        project_root=project_root,
        dataset_name=dataset,
        row_limit=nrows,
    )
    probe_config = ProbeConfig(max_train_rows=max_train_rows)
    split_config = SplitConfig()
    # Use PaySim probe for PaySim dataset, otherwise use the default financial fraud detection probe
    if "paysim" in dataset.lower():
        PaySimSignalProbe(
            dataset_config=dataset_config,
            probe_config=probe_config,
            split_config=split_config,
        ).run()
    else:
        SignalProbe(
            dataset_config=dataset_config,
            probe_config=probe_config,
            split_config=split_config,
        ).run()
    return 0

def _run_ablation(project_root: Path, dataset: str, nrows: int | None, no_account_code: bool) -> int:
    """Execute the full/no-history/history-only ablation baselines."""
    dataset_config = DatasetConfig(project_root=project_root, dataset_name=dataset, row_limit=nrows)
    split_config = SplitConfig()
    BaselineEvaluator(
        dataset_config=dataset_config,
        split_config=split_config,
        keep_account_code=not no_account_code,
    ).run()
    return 0


def _run_summarize_results(project_root: Path, dataset: str, out: Path | None, legacy_since: str) -> int:
    """Summarize existing reports into markdown result tables."""
    dataset_config = DatasetConfig(project_root=project_root, dataset_name=dataset)
    ResultsSummarizer(
        dataset_config=dataset_config,
        summary_config=ResultsSummaryConfig(legacy_since=legacy_since, output_path=out),
    ).run()
    return 0


def _run_train_model(
    project_root: Path,
    dataset: str,
    nrows: int | None,
    window: int,
    bottleneck: bool,
    plain: bool,
    attention_pooling: bool,
    seeds: list[int] | None,
) -> int:
    """Execute LSTM-Autoencoder training, as a single run or a multi-seed aggregate."""
    if sum([bottleneck, plain, attention_pooling]) > 1:
        raise ValueError(
            "--bottleneck, --plain, and --attention-pooling are mutually exclusive. "
            "Choose one architecture or neither (default: attention)."
            )
    config = DatasetConfig(project_root=project_root, dataset_name=dataset, row_limit=nrows)
    sequence_config = SequenceConfig(window_length=window)
    pipeline = PaySimModelTrainingPipeline(
        dataset_config=config, sequence_config=sequence_config,
        model_config=ModelConfig(
            bottleneck=bottleneck,
            attention=not plain and not bottleneck and not attention_pooling,
            attention_pooling=attention_pooling,
        ),
        train_config=TrainConfig(),
        split_config=SplitConfig(),
    )
    if seeds:
        pipeline.run_with_seeds(seeds)
    else:
        pipeline.run()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _run_probe_hybrid_signal(
    project_root: Path,
    dataset: str,
    nrows: int | None,
    window: int,
    bottleneck: bool,
    plain: bool,
    attention_pooling: bool,
    seed: int,
    checkpoint: Path | None,
) -> int:
    """Execute the LSTM-AE-vs-LR correlation probe."""
    if sum([bottleneck, plain, attention_pooling]) > 1:
        raise ValueError(
            "--bottleneck, --plain, and --attention-pooling are mutually exclusive. "
            "Choose one architecture or neither (default: attention)."
            )
    config = DatasetConfig(project_root=project_root, dataset_name=dataset, row_limit=nrows)
    sequence_config = SequenceConfig(window_length=window)
    HybridSignalProbe(
        dataset_config=config,
        sequence_config=sequence_config,
        model_config=ModelConfig(
            bottleneck=bottleneck,
            attention=not plain and not bottleneck and not attention_pooling,
            attention_pooling=attention_pooling,
        ),
        split_config=SplitConfig(),
        checkpoint_path=checkpoint,
        seed=seed,
    ).run()
    return 0


def _run_eval_hybrid(
    project_root: Path,
    dataset: str,
    nrows: int | None,
    window: int,
    bottleneck: bool,
    plain: bool,
    attention_pooling: bool,
    seeds: list[int],
    n_bootstrap: int,
) -> int:
    """Execute the P3.5 hybrid classifier evaluation on already-trained checkpoints."""
    if sum([bottleneck, plain, attention_pooling]) > 1:
        raise ValueError(
            "--bottleneck, --plain, and --attention-pooling are mutually exclusive. "
            "Choose one architecture or neither (default: attention)."
            )
    config = DatasetConfig(project_root=project_root, dataset_name=dataset, row_limit=nrows)
    HybridClassifierEvaluator(
        dataset_config=config,
        sequence_config=SequenceConfig(window_length=window),
        model_config=ModelConfig(
            bottleneck=bottleneck,
            attention=not plain and not bottleneck and not attention_pooling,
            attention_pooling=attention_pooling,
        ),
        config=HybridClassifierConfig(seeds=tuple(seeds), n_bootstrap=n_bootstrap),
    ).run()
    return 0


def _run_eval_unsupervised(
    project_root: Path,
    dataset: str,
    nrows: int | None,
    seeds: list[int],
    keep_account_code: bool,
    latent_dim: int,
    hidden_dim: int,
) -> int:
    """Execute the RQ D1 unsupervised baselines (Isolation Forest, dense AE)."""
    config = DatasetConfig(project_root=project_root, dataset_name=dataset, row_limit=nrows)
    UnsupervisedBaselineEvaluator(
        dataset_config=config,
        dense_config=DenseAutoencoderConfig(hidden_dim=hidden_dim, latent_dim=latent_dim),
        config=UnsupervisedBaselineConfig(
            seeds=tuple(seeds),
            drop_columns=() if keep_account_code else UnsupervisedBaselineConfig.drop_columns,
        ),
    ).run()
    return 0


def _run_train_all_configs(
    project_root: Path,
    dataset: str,
    nrows: int | None,
    windows: str,
    seeds: str,
    architectures: str,
    force: bool,
    dry_run: bool,
) -> int:
    """Train LSTM-AE for a grid of windows, seeds, and architectures."""
    import subprocess
    from itertools import product

    VALID_ARCHITECTURES = ["attention", "bottleneck", "plain", "attention_pooling"]

    windows_list = [int(w.strip()) for w in windows.split(",")]
    seeds_list = [int(s.strip()) for s in seeds.split(",")]
    arch_list = [a.strip() for a in architectures.split(",")]

    # Validate architectures
    unknown = [a for a in arch_list if a not in VALID_ARCHITECTURES]
    if unknown:
        raise ValueError(f"Unknown architecture(s): {unknown}. Choose from: {VALID_ARCHITECTURES}")

    seed_list_str = ",".join(map(str, seeds_list))
    total_configs = len(windows_list) * len(arch_list)

    print("Starting training for selected configurations")
    print(f"Dataset: {dataset}")
    print(f"Windows: {windows_list}")
    print(f"Seeds: {seeds_list}")
    print(f"Architectures: {arch_list}")
    print(f"Force retrain: {force}")
    if dry_run:
        print("Dry run: nothing will be trained")
    print(f"Total configurations: {total_configs} (x {len(seeds_list)} seeds each)")
    print(f"Project root: {project_root}")
    print()

    current = 0
    skipped = 0
    completed = 0
    failed = 0

    for window, arch_name in product(windows_list, arch_list):
        current += 1

        # Get architecture flags
        if arch_name == "attention":
            arch_flags = []
        elif arch_name == "bottleneck":
            arch_flags = ["--bottleneck"]
        elif arch_name == "plain":
            arch_flags = ["--plain"]
        elif arch_name == "attention_pooling":
            arch_flags = ["--attention-pooling"]
        else:
            arch_flags = []

        # Check if checkpoints exist
        arch_tag = {
            "attention": "attn",
            "bottleneck": "bottleneck",
            "plain": "plain",
            "attention_pooling": "attn_pool",
        }[arch_name]

        all_exist = True
        for seed in seeds_list:
            checkpoint_name = f"lstm_ae_w{window}_{arch_tag}_seed{seed}_best.pt"
            checkpoint_path = project_root / "artifacts" / "model" / "paysim_synthetic_financial" / checkpoint_name
            if not checkpoint_path.exists():
                all_exist = False
                break

        # Skip if all checkpoints exist and not forcing
        if not force and all_exist:
            print(f"[{current}/{total_configs}] Skipping: Window={window}, Architecture={arch_name}")
            print("  Checkpoints already exist, skipping (use --force to retrain)...")
            for seed in seeds_list:
                checkpoint_name = f"lstm_ae_w{window}_{arch_tag}_seed{seed}_best.pt"
                print(f"  - {checkpoint_name}")
            print()
            skipped += 1
            continue

        print(f"[{current}/{total_configs}] Training: Window={window}, Architecture={arch_name}, Seeds={seed_list_str}")
        print()

        cmd = [
            "uv", "run", "python", "-m", "lstm_autoencoder", "train-model",
            "--dataset", dataset,
            "--window", str(window),
        ] + arch_flags + ["--seeds", seed_list_str]

        if nrows:
            cmd.extend(["--nrows", str(nrows)])

        if dry_run:
            print(f"Would run: {' '.join(cmd)}")
            print()
            continue

        try:
            result = subprocess.run(cmd, check=True)
            if result.returncode == 0:
                print(f"Completed: Window={window}, Architecture={arch_name}")
                print()
                completed += 1
            else:
                print(f"Error: Command failed with exit code {result.returncode}")
                print(f"Command: {' '.join(cmd)}")
                print()
                failed += 1
        except Exception as e:
            print(f"Error: Command failed with exception: {e}")
            print(f"Command: {' '.join(cmd)}")
            print()
            failed += 1

    print("Training Summary")
    print(f"Total configurations: {total_configs}")
    print(f"Completed: {completed}")
    print(f"Skipped (already exists): {skipped}")
    if failed > 0:
        print(f"Failed: {failed}")
    print()

    if failed == 0:
        print("Selected training runs completed successfully!")
    else:
        print(f"Training completed with {failed} failures.")

    return 0 if failed == 0 else 1


def _run_eval_hybrid_all_configs(
    project_root: Path,
    dataset: str,
    nrows: int | None,
    windows: str,
    seeds: str,
    architectures: str,
    force: bool,
    dry_run: bool,
    bootstrap: int,
) -> int:
    """Evaluate hybrid classifier for a grid of windows, seeds, and architectures."""
    import subprocess
    from itertools import product

    VALID_ARCHITECTURES = ["attention", "bottleneck", "plain", "attention_pooling"]

    windows_list = [int(w.strip()) for w in windows.split(",")]
    seeds_list = [int(s.strip()) for s in seeds.split(",")]
    arch_list = [a.strip() for a in architectures.split(",")]

    # Validate architectures
    unknown = [a for a in arch_list if a not in VALID_ARCHITECTURES]
    if unknown:
        raise ValueError(f"Unknown architecture(s): {unknown}. Choose from: {VALID_ARCHITECTURES}")

    seed_list_str = ",".join(map(str, seeds_list))
    total_configs = len(windows_list) * len(arch_list)

    print("Starting hybrid classifier evaluation for selected configurations")
    print(f"Dataset: {dataset}")
    print(f"Windows: {windows_list}")
    print(f"Seeds: {seeds_list}")
    print(f"Architectures: {arch_list}")
    print(f"Force re-evaluate: {force}")
    print(f"Bootstrap samples: {bootstrap}")
    if nrows:
        print(f"Row limit: {nrows}")
    if dry_run:
        print("Dry run: nothing will be evaluated")
    print(f"Total configurations: {total_configs}")
    print(f"Project root: {project_root}")
    print()

    current = 0
    skipped = 0
    completed = 0
    failed = 0

    for window, arch_name in product(windows_list, arch_list):
        current += 1

        # Get architecture flags
        if arch_name == "attention":
            arch_flags = []
        elif arch_name == "bottleneck":
            arch_flags = ["--bottleneck"]
        elif arch_name == "plain":
            arch_flags = ["--plain"]
        elif arch_name == "attention_pooling":
            arch_flags = ["--attention-pooling"]
        else:
            arch_flags = []

        # Check if report exists
        report_dir = project_root / "artifacts" / "report" / "paysim_synthetic_financial" / "hybrid_classifier"
        report_exists = False
        if report_dir.exists():
            for timestamp_dir in report_dir.iterdir():
                if timestamp_dir.is_dir():
                    for config_dir in timestamp_dir.iterdir():
                        if config_dir.is_dir():
                            expected_pattern = f"window={window}_architecture={arch_name}"
                            if config_dir.name == expected_pattern:
                                json_path = config_dir / "hybrid_classifier_report.json"
                                if json_path.exists():
                                    report_exists = True
                                    break
                if report_exists:
                    break

        # Skip if report exists and not forcing
        if not force and report_exists:
            print(f"[{current}/{total_configs}] Skipping: Window={window}, Architecture={arch_name}")
            print("  Report already exists, skipping (use --force to re-evaluate)...")
            print()
            skipped += 1
            continue

        print(
            f"[{current}/{total_configs}] Evaluating: Window={window}, Architecture={arch_name}, Seeds={seed_list_str}"
        )
        print()

        cmd = [
            "uv", "run", "python", "-m", "lstm_autoencoder", "eval-hybrid",
            "--dataset", dataset,
            "--window", str(window),
        ] + arch_flags + ["--seeds", seed_list_str, "--bootstrap", str(bootstrap)]

        if nrows:
            cmd.extend(["--nrows", str(nrows)])

        if dry_run:
            print(f"Would run: {' '.join(cmd)}")
            print()
            continue

        try:
            result = subprocess.run(cmd, check=True)
            if result.returncode == 0:
                print(f"Completed: Window={window}, Architecture={arch_name}")
                print()
                completed += 1
            else:
                print(f"Error: Command failed with exit code {result.returncode}")
                print(f"Command: {' '.join(cmd)}")
                print()
                failed += 1
        except Exception as e:
            print(f"Error: Command failed with exception: {e}")
            print(f"Command: {' '.join(cmd)}")
            print()
            failed += 1

    print("Evaluation Summary")
    print(f"Total configurations: {total_configs}")
    print(f"Completed: {completed}")
    print(f"Skipped (already exists): {skipped}")
    if failed > 0:
        print(f"Failed: {failed}")
    print()

    if failed == 0:
        print("Selected evaluation runs completed successfully!")
    else:
        print(f"Evaluation completed with {failed} failures.")

    return 0 if failed == 0 else 1


def _run_pipeline(
    project_root: Path,
    dataset: str,
    nrows: int | None,
    windows: str,
    seeds: str,
    architectures: str,
    bootstrap: int,
) -> int:
    """Run end-to-end pipeline: hybrid evaluation, unsupervised evaluation, summarize."""
    import subprocess

    windows_list = [int(w.strip()) for w in windows.split(",")]

    print("Running end-to-end pipeline")
    print(f"Dataset: {dataset}")
    print(f"Windows: {windows_list}")
    print(f"Seeds: {seeds}")
    print(f"Architectures: {architectures}")
    print(f"Bootstrap: {bootstrap}")
    print()

    # Step 1: Evaluate Hybrid Classifier for all configurations
    for window in windows_list:
        print(f"Evaluating hybrid for window {window}...")
        cmd = [
            "uv", "run", "python", "-m", "lstm_autoencoder", "eval-hybrid-all-configs",
            "--dataset", dataset,
            "--windows", str(window),
            "--seeds", seeds,
            "--architectures", architectures,
            "--bootstrap", str(bootstrap),
        ]
        if nrows:
            cmd.extend(["--nrows", str(nrows)])

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Hybrid evaluation failed for window {window}")
            return 1

    # Step 2: Evaluate Unsupervised Baselines
    print("Evaluating unsupervised baselines...")
    cmd = [
        "uv", "run", "python", "-m", "lstm_autoencoder", "eval-unsupervised",
        "--dataset", dataset,
        "--seeds", seeds,
    ]
    if nrows:
        cmd.extend(["--nrows", str(nrows)])

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Unsupervised evaluation failed")
        return 1

    # Step 3: Summarize Results
    print("Summarizing results...")
    cmd = [
        "uv", "run", "python", "-m", "lstm_autoencoder", "summarize-results",
        "--dataset", dataset,
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Summarization failed")
        return 1

    print("Pipeline completed successfully!")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the selected CLI command."""
    parser = build_parser()
    arguments = parser.parse_args(argv)

    project_root = arguments.project_root.resolve()

    commands = {
        "inspect-data": lambda: _run_inspect_data(
            project_root, arguments.dataset, arguments.nrows
        ),
        "probe-signal": lambda: _run_probe_signal(
            project_root,
            arguments.dataset,
            arguments.nrows,
            arguments.max_train_rows,
        ),
        "ablation": lambda: _run_ablation(
            project_root, arguments.dataset, arguments.nrows, arguments.no_account_code
        ),
        "train-model": lambda: _run_train_model(
            project_root,
            arguments.dataset,
            arguments.nrows,
            arguments.window,
            arguments.bottleneck,
            arguments.plain,
            arguments.attention_pooling,
            [int(s) for s in arguments.seeds.split(",")] if arguments.seeds else None,
        ),
        "eval-hybrid": lambda: _run_eval_hybrid(
            project_root,
            arguments.dataset,
            arguments.nrows,
            arguments.window,
            arguments.bottleneck,
            arguments.plain,
            arguments.attention_pooling,
            [int(seed) for seed in arguments.seeds.split(",")],
            arguments.bootstrap,
        ),
        "eval-unsupervised": lambda: _run_eval_unsupervised(
            project_root,
            arguments.dataset,
            arguments.nrows,
            [int(seed) for seed in arguments.seeds.split(",")],
            arguments.keep_account_code,
            arguments.latent_dim,
            arguments.hidden_dim,
        ),
        "summarize-results": lambda: _run_summarize_results(
            project_root, arguments.dataset, arguments.out, arguments.legacy_since
        ),
        "probe-hybrid-signal": lambda: _run_probe_hybrid_signal(
            project_root,
            arguments.dataset,
            arguments.nrows,
            arguments.window,
            arguments.bottleneck,
            arguments.plain,
            arguments.attention_pooling,
            arguments.seed,
            arguments.checkpoint,
        ),
        "train-all-configs": lambda: _run_train_all_configs(
            project_root,
            arguments.dataset,
            arguments.nrows,
            arguments.windows,
            arguments.seeds,
            arguments.architectures,
            arguments.force,
            arguments.dry_run,
        ),
        "eval-hybrid-all-configs": lambda: _run_eval_hybrid_all_configs(
            project_root,
            arguments.dataset,
            arguments.nrows,
            arguments.windows,
            arguments.seeds,
            arguments.architectures,
            arguments.force,
            arguments.dry_run,
            arguments.bootstrap,
        ),
        "run-pipeline": lambda: _run_pipeline(
            project_root,
            arguments.dataset,
            arguments.nrows,
            arguments.windows,
            arguments.seeds,
            arguments.architectures,
            arguments.bootstrap,
        ),
    }

    try:
        handler = commands.get(arguments.command)

        if handler is None:
            parser.error(f"Unknown command: {arguments.command}")

        return handler()

    except (OSError, ValueError) as error:
        parser.error(str(error))

    return 2