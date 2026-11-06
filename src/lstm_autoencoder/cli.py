"""Command-line entry points for the LSTM Autoencoder research project."""

from __future__ import annotations

import argparse
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
            "repeated across the window). Mutually exclusive with --plain."
        ),
    )
    train_parser.add_argument(
        "--plain",
        action="store_true",
        help=(
            "Use raw per-step encoder outputs (can copy directly). "
            "Mutually exclusive with --bottleneck. Default is attention."
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
        help="Match the checkpoint's architecture: bottleneck decoder. Mutually exclusive with --plain.",
    )
    hybrid_parser.add_argument(
        "--plain",
        action="store_true",
        help="Match the checkpoint's architecture: raw per-step outputs. Mutually exclusive with --bottleneck.",
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
        help="Match the checkpoints' architecture: bottleneck decoder. Mutually exclusive with --plain.",
    )
    eval_hybrid_parser.add_argument(
        "--plain",
        action="store_true",
        help="Match the checkpoints' architecture: raw per-step outputs. Mutually exclusive with --bottleneck.",
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

def _run_ablation(project_root: Path, dataset: str, nrows: int | None) -> int:
    """Execute the full/no-history/history-only ablation baselines."""
    dataset_config = DatasetConfig(project_root=project_root, dataset_name=dataset, row_limit=nrows)
    split_config = SplitConfig()
    BaselineEvaluator(dataset_config=dataset_config, split_config=split_config).run()
    return 0


def _run_train_model(
    project_root: Path,
    dataset: str,
    nrows: int | None,
    window: int,
    bottleneck: bool,
    plain: bool,
    seeds: list[int] | None,
) -> int:
    """Execute LSTM-Autoencoder training, as a single run or a multi-seed aggregate."""
    if bottleneck and plain:
        raise ValueError(
            "--bottleneck and --plain are mutually exclusive. Choose one architecture or neither (default: attention)."
            )
    config = DatasetConfig(project_root=project_root, dataset_name=dataset, row_limit=nrows)
    sequence_config = SequenceConfig(window_length=window)
    pipeline = PaySimModelTrainingPipeline(
        dataset_config=config, sequence_config=sequence_config,
        model_config=ModelConfig(bottleneck=bottleneck, attention=not plain and not bottleneck),
        train_config=TrainConfig(),
    )
    if seeds:
        pipeline.run_with_seeds(seeds)
    else:
        pipeline.run()
    return 0


def _run_probe_hybrid_signal(
    project_root: Path,
    dataset: str,
    nrows: int | None,
    window: int,
    bottleneck: bool,
    plain: bool,
    seed: int,
    checkpoint: Path | None,
) -> int:
    """Execute the LSTM-AE-vs-LR correlation probe."""
    if bottleneck and plain:
        raise ValueError(
            "--bottleneck and --plain are mutually exclusive. Choose one architecture or neither (default: attention)."
            )
    config = DatasetConfig(project_root=project_root, dataset_name=dataset, row_limit=nrows)
    sequence_config = SequenceConfig(window_length=window)
    HybridSignalProbe(
        dataset_config=config,
        sequence_config=sequence_config,
        model_config=ModelConfig(bottleneck=bottleneck, attention=not plain and not bottleneck),
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
    seeds: list[int],
    n_bootstrap: int,
) -> int:
    """Execute the P3.5 hybrid classifier evaluation on already-trained checkpoints."""
    if bottleneck and plain:
        raise ValueError(
            "--bottleneck and --plain are mutually exclusive. Choose one architecture or neither (default: attention)."
            )
    config = DatasetConfig(project_root=project_root, dataset_name=dataset, row_limit=nrows)
    HybridClassifierEvaluator(
        dataset_config=config,
        sequence_config=SequenceConfig(window_length=window),
        model_config=ModelConfig(bottleneck=bottleneck, attention=not plain and not bottleneck),
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
            project_root, arguments.dataset, arguments.nrows
        ),
        "train-model": lambda: _run_train_model(
            project_root,
            arguments.dataset,
            arguments.nrows,
            arguments.window,
            arguments.bottleneck,
            arguments.plain,
            [int(s) for s in arguments.seeds.split(",")] if arguments.seeds else None,
        ),
        "eval-hybrid": lambda: _run_eval_hybrid(
            project_root,
            arguments.dataset,
            arguments.nrows,
            arguments.window,
            arguments.bottleneck,
            arguments.plain,
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
        "probe-hybrid-signal": lambda: _run_probe_hybrid_signal(
            project_root,
            arguments.dataset,
            arguments.nrows,
            arguments.window,
            arguments.bottleneck,
            arguments.plain,
            arguments.seed,
            arguments.checkpoint,
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