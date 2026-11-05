"""Command-line entry points for the LSTM Autoencoder research project."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from lstm_autoencoder.config import DatasetConfig, ProbeConfig
from lstm_autoencoder.data.financial_fraud_detection.exploratory import FinancialFraudDetectionExplorer
from lstm_autoencoder.data.paysim_synthetic_financial.exploratory import PaySimSyntheticFinancialDatasetsExplorer
from lstm_autoencoder.evaluation.financial_fraud_detection.signal_probe import SignalProbe
from lstm_autoencoder.evaluation.paysim_synthetic_financial.baseline import BaselineEvaluator
from lstm_autoencoder.evaluation.paysim_synthetic_financial.signal_probe import (
    PaySimSignalProbe,
)
from lstm_autoencoder.model.config import ModelConfig, SequenceConfig, TrainConfig
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
        help="Train the hybrid LSTM-Autoencoder with attention on PaySim.",
    )
    _add_dataset_arguments(train_parser)
    train_parser.add_argument("--window", type=int, default=SequenceConfig().window_length)
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
    config = DatasetConfig(
        project_root=project_root,
        dataset_name=dataset,
        row_limit=nrows,
    )
    explorers = {
        "financial_fraud_detection": FinancialFraudDetectionExplorer,
        "paysim_synthetic_financial": PaySimSyntheticFinancialDatasetsExplorer,
    }
    explorer = explorers.get(dataset.lower())
    if explorer is None:
        return 0
    explorer(config=config).run()
    return 0


def _run_probe_signal(
    project_root: Path, dataset: str, nrows: int | None, max_train_rows: int
) -> int:
    """Execute the signal learnability probe."""
    config = DatasetConfig(
        project_root=project_root,
        dataset_name=dataset,
        row_limit=nrows,
    )
    probe_config = ProbeConfig(max_train_rows=max_train_rows)
    # Use PaySim probe for PaySim dataset, otherwise use the default financial fraud detection probe
    if "paysim" in dataset.lower():
        PaySimSignalProbe(config=config, probe_config=probe_config).run()
    else:
        SignalProbe(config=config, probe_config=probe_config).run()
    return 0

def _run_ablation(project_root: Path, dataset: str, nrows: int | None) -> int:
    """Execute the full/no-history/history-only ablation baselines."""
    config = DatasetConfig(project_root=project_root, dataset_name=dataset, row_limit=nrows)
    BaselineEvaluator(config=config).run()
    return 0


def _run_train_model(
    project_root: Path, dataset: str, nrows: int | None, window: int, seeds: list[int] | None
) -> int:
    """Execute LSTM-Autoencoder training, as a single run or a multi-seed aggregate."""
    config = DatasetConfig(project_root=project_root, dataset_name=dataset, row_limit=nrows)
    sequence_config = SequenceConfig(window_length=window)
    pipeline = PaySimModelTrainingPipeline(
        dataset_config=config, sequence_config=sequence_config,
        model_config=ModelConfig(), train_config=TrainConfig(),
    )
    if seeds:
        pipeline.run_with_seeds(seeds)
    else:
        pipeline.run()
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
            [int(s) for s in arguments.seeds.split(",")] if arguments.seeds else None,
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