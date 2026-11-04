"""Command-line entry points for the LSTM Autoencoder research project."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from lstm_autoencoder.config import DatasetConfig, ProbeConfig
from lstm_autoencoder.data.financial_fraud_detection.exploratory import FinancialFraudDetectionExplorer
from lstm_autoencoder.data.paysim_synthetic_financial.exploratory import PaySimSyntheticFinancialDatasetsExplorer
from lstm_autoencoder.evaluation.financial_fraud_detection.signal_probe import SignalProbe
from lstm_autoencoder.evaluation.paysim_synthetic_financial.signal_probe import (
    PaySimSignalProbe,
)


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


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the selected CLI command."""
    parser = build_parser()
    arguments = parser.parse_args(argv)

    project_root = arguments.project_root.resolve()

    try:
        if arguments.command == "inspect-data":
            return _run_inspect_data(
                project_root, arguments.dataset, arguments.nrows
            )
        if arguments.command == "probe-signal":
            return _run_probe_signal(
                project_root,
                arguments.dataset,
                arguments.nrows,
                arguments.max_train_rows
            )

        parser.error(f"Unknown command: {arguments.command}")

    except (OSError, ValueError) as error:
        parser.error(str(error))

    return 2
