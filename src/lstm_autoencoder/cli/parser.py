"""Parser for the LSTM Autoencoder research project CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from lstm_autoencoder.evaluation.paysim_synthetic_financial.hybrid_classifier import HybridClassifierConfig
from lstm_autoencoder.evaluation.paysim_synthetic_financial.results_summary import DEFAULT_LEGACY_SINCE
from lstm_autoencoder.evaluation.paysim_synthetic_financial.score_diagnostics import (
    SPLITS as DIAGNOSTIC_SPLITS,
)
from lstm_autoencoder.evaluation.paysim_synthetic_financial.score_diagnostics import ScoreDiagnosticsConfig
from lstm_autoencoder.evaluation.paysim_synthetic_financial.signal_probe import ProbeConfig
from lstm_autoencoder.model.config import (
    SELECTION_METRICS,
    DenseAutoencoderConfig,
    ModelConfig,
    SequenceConfig,
    TrainConfig,
)


class ParserArgs:
    """Parser for CLI arguments."""

    @staticmethod
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
    
    @staticmethod
    def _add_capacity_arguments(parser: argparse.ArgumentParser) -> None:
        """Add the model-capacity flags. They also change the checkpoint name, so every
        command that loads a checkpoint must be given the same values it was trained with."""
        parser.add_argument(
            "--hidden-dim",
            type=int,
            default=ModelConfig.hidden_dim,
            help="LSTM hidden size. A non-default value adds an `_h{N}` suffix to the architecture tag.",
        )
        parser.add_argument(
            "--bottleneck-dim",
            type=int,
            default=None,
            help=(
                "Size of the latent code the single-vector context is squeezed through "
                "(needs --bottleneck or --attention-pooling; adds a `_z{N}` suffix). "
                "Keep it well below the 5 x window input values, otherwise the model can learn the identity."
            ),
        )

    @staticmethod
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
        ParserArgs._add_dataset_arguments(inspect_parser)

        probe_parser = subparsers.add_parser(
            "probe-signal",
            help="Check whether the fraud label can be learned from the features.",
        )
        ablation_parser = subparsers.add_parser(
            "ablation",
            help="Train full/no-history/history-only baselines and compare per history bucket.",
        )
        ParserArgs._add_dataset_arguments(ablation_parser)
        ablation_parser.add_argument(
            "--no-account-code",
            action="store_true",
            help="Exclude account_code from features (it's a factorized ID, not behavioral).",
        )

        train_parser = subparsers.add_parser(
            "train-model",
            help="Train the LSTM-Autoencoder on PaySim with configurable architecture.",
        )
        ParserArgs._add_dataset_arguments(train_parser)
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
        ParserArgs._add_capacity_arguments(train_parser)
        train_parser.add_argument(
            "--selection-metric",
            choices=SELECTION_METRICS,
            default=TrainConfig.selection_metric,
            help=(
                "Checkpoint selection: val_loss = mean reconstruction error of NORMAL validation rows "
                "(label-free, default); val_auc = reconstruction-error AUC on labelled validation rows "
                "(uses labels: state this selection bias in the paper)."
            ),
        )
        train_parser.add_argument(
            "--max-epochs", type=int, default=TrainConfig.max_epochs, help="Maximum training epochs."
        )
        train_parser.add_argument(
            "--patience",
            type=int,
            default=TrainConfig.patience,
            help="Epochs without a significant improvement before stopping.",
        )
        train_parser.add_argument(
            "--min-delta",
            type=float,
            default=TrainConfig.min_delta,
            help="Relative improvement of the selection metric that counts as progress (0.01 = 1%%).",
        )
        hybrid_parser = subparsers.add_parser(
            "probe-hybrid-signal",
            help=(
                "Correlate a trained LSTM-AE checkpoint's reconstruction error with "
                "the LR baseline's score on the same test rows, before retraining a hybrid model."
            ),
        )
        ParserArgs._add_dataset_arguments(hybrid_parser)
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
        ParserArgs._add_capacity_arguments(hybrid_parser)
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
        ParserArgs._add_dataset_arguments(eval_hybrid_parser)
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
        ParserArgs._add_capacity_arguments(eval_hybrid_parser)
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
        eval_hybrid_parser.add_argument(
            "--bootstrap-mode",
            type=str,
            choices=["row", "receiver"],
            default=HybridClassifierConfig.bootstrap_mode,
            help=(
                "'receiver' (default): resample whole receivers, so correlated rows from the "
                "same receiver move together (P5.1 cluster bootstrap). 'row': legacy class-"
                "stratified row resampling, ignores within-receiver correlation."
            ),
        )

        eval_unsupervised_parser = subparsers.add_parser(
            "eval-unsupervised",
            help=(
                "RQ D1 baselines: Isolation Forest (all features and the 5 sequence features) and a "
                "dense autoencoder on the same split, scored on the same test rows."
            ),
        )
        ParserArgs._add_dataset_arguments(eval_unsupervised_parser)
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

        diagnose_parser = subparsers.add_parser(
            "diagnose-scores",
            help=(
                "P2.0/P2.1: compare mean/last/max reconstruction scores and test whether the error "
                "tracks history length, on already-trained checkpoints (no retraining)."
            ),
        )
        ParserArgs._add_dataset_arguments(diagnose_parser)
        diagnose_parser.add_argument("--window", type=int, default=SequenceConfig().window_length)
        diagnose_parser.add_argument(
            "--bottleneck",
            action="store_true",
            help=(
                "Match the checkpoints' architecture: bottleneck decoder."
                "Mutually exclusive with --plain and --attention-pooling."
            )
        )
        diagnose_parser.add_argument(
            "--plain",
            action="store_true",
            help=(
                "Match the checkpoints' architecture: raw per-step outputs."
                "Mutually exclusive with --bottleneck and --attention-pooling."
            )
        )
        diagnose_parser.add_argument(
            "--attention-pooling",
            action="store_true",
            help=(
                "Match the checkpoints' architecture: attention pooling"
                "Mutually exclusive with --bottleneck and --plain."
            )
            ,
        )
        ParserArgs._add_capacity_arguments(diagnose_parser)
        diagnose_parser.add_argument(
            "--seeds",
            type=str,
            default="42",
            help="Comma-separated seeds of already-trained checkpoints (e.g. 42,123,2026).",
        )
        diagnose_parser.add_argument(
            "--split",
            choices=DIAGNOSTIC_SPLITS,
            default=ScoreDiagnosticsConfig.split,
            help=(
                "Split to score. Defaults to val so design decisions never use test; "
                "test additionally reports the LR-outcome groups."
            ),
        )

        train_all_parser = subparsers.add_parser(
            "train-all-configs",
            help="Train LSTM-AE for a grid of windows, seeds, and architectures.",
        )
        ParserArgs._add_dataset_arguments(train_all_parser)
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
            help=(
                "Comma-separated architectures (default: all)"
                "Choices: attention, bottleneck, plain, attention_pooling."
            ),
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
        ParserArgs._add_dataset_arguments(eval_hybrid_all_parser)
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
            help=(
                "Comma-separated architectures (default: all)."
                "Choices: attention, bottleneck, plain, attention_pooling."
            ),
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
            default=HybridClassifierConfig.n_bootstrap,
            help="Paired bootstrap resamples per seed for the delta CIs (0 to skip).",
        )
        eval_hybrid_all_parser.add_argument(
            "--bootstrap-mode",
            type=str,
            choices=["row", "receiver"],
            default=HybridClassifierConfig.bootstrap_mode,
            help="'receiver' (default) or 'row' -- see `eval-hybrid --bootstrap-mode`.",
        )

        run_pipeline_parser = subparsers.add_parser(
            "run-pipeline",
            help="Run end-to-end pipeline: hybrid evaluation, unsupervised evaluation, summarize.",
        )
        ParserArgs._add_dataset_arguments(run_pipeline_parser)
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
            help=(
                "Comma-separated architectures (default: all)"
                "Choices: attention, bottleneck, plain, attention_pooling."
            ),
        )
        run_pipeline_parser.add_argument(
            "--bootstrap",
            type=int,
            default=HybridClassifierConfig.n_bootstrap,
            help="Paired bootstrap resamples per seed for the delta CIs (0 to skip).",
        )
        run_pipeline_parser.add_argument(
            "--bootstrap-mode",
            type=str,
            choices=["row", "receiver"],
            default=HybridClassifierConfig.bootstrap_mode,
            help="'receiver' (default) or 'row' -- see `eval-hybrid --bootstrap-mode`.",
        )

        ParserArgs._add_dataset_arguments(probe_parser)
        probe_parser.add_argument(
            "--max-train-rows",
            type=int,
            default=ProbeConfig.max_train_rows,
            help="Maximum number of training rows used by the probe model.",
        )
        return parser

