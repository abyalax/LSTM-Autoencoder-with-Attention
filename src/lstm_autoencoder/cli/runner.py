"""Runner CLI for the research project"""

from __future__ import annotations

import os
import sys
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
from lstm_autoencoder.evaluation.paysim_synthetic_financial.pipeline_audit import (
    PipelineAudit,
    PipelineAuditConfig,
)
from lstm_autoencoder.evaluation.paysim_synthetic_financial.results_summary import (
    ResultsSummarizer,
    ResultsSummaryConfig,
)
from lstm_autoencoder.evaluation.paysim_synthetic_financial.score_diagnostics import (
    ScoreDiagnostics,
    ScoreDiagnosticsConfig,
)
from lstm_autoencoder.evaluation.paysim_synthetic_financial.score_strategy import (
    ScoreStrategyConfig,
    ScoreStrategySelector,
)
from lstm_autoencoder.evaluation.paysim_synthetic_financial.signal_probe import (
    PaySimSignalProbe,
)
from lstm_autoencoder.evaluation.paysim_synthetic_financial.standalone_evaluator import (
    ARCHITECTURES as STANDALONE_ARCHITECTURES,
)
from lstm_autoencoder.evaluation.paysim_synthetic_financial.standalone_evaluator import (
    StandaloneConfig,
    StandaloneEvaluator,
)
from lstm_autoencoder.evaluation.paysim_synthetic_financial.unsupervised_baselines import (
    UnsupervisedBaselineConfig,
    UnsupervisedBaselineEvaluator,
)
from lstm_autoencoder.model.config import (
    DenseAutoencoderConfig,
    ModelConfig,
    SequenceConfig,
    TrainConfig,
)
from lstm_autoencoder.model.pipeline import PaySimModelTrainingPipeline
from lstm_autoencoder.reporting.figures import FigureBuilder


class RunnerCLI:
    """Runner for CLI commands."""

    @staticmethod
    def run_inspect_data(project_root: Path, dataset: str, nrows: int | None) -> int:
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

    @staticmethod
    def run_probe_signal(project_root: Path, dataset: str, nrows: int | None, max_train_rows: int) -> int:
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

    @staticmethod
    def run_ablation(project_root: Path, dataset: str, nrows: int | None, no_account_code: bool) -> int:
        """Execute the full/no-history/history-only ablation baselines."""
        dataset_config = DatasetConfig(project_root=project_root, dataset_name=dataset, row_limit=nrows)
        split_config = SplitConfig()
        BaselineEvaluator(
            dataset_config=dataset_config,
            split_config=split_config,
            keep_account_code=not no_account_code,
        ).run()
        return 0

    @staticmethod
    def run_summarize_results(project_root: Path, dataset: str, out: Path | None, legacy_since: str) -> int:
        """Summarize existing reports into markdown result tables."""
        dataset_config = DatasetConfig(project_root=project_root, dataset_name=dataset)
        ResultsSummarizer(
            dataset_config=dataset_config,
            summary_config=ResultsSummaryConfig(legacy_since=legacy_since, output_path=out),
        ).run()
        return 0

    @staticmethod
    def run_train_model(
        project_root: Path,
        dataset: str,
        nrows: int | None,
        window: int,
        model_config: ModelConfig,
        train_config: TrainConfig,
        seeds: list[int] | None,
    ) -> int:
        """Execute LSTM-Autoencoder training, as a single run or a multi-seed aggregate."""
        config = DatasetConfig(project_root=project_root, dataset_name=dataset, row_limit=nrows)
        sequence_config = SequenceConfig(window_length=window)
        pipeline = PaySimModelTrainingPipeline(
            dataset_config=config,
            sequence_config=sequence_config,
            model_config=model_config,
            train_config=train_config,
            split_config=SplitConfig(),
        )
        if seeds:
            pipeline.run_with_seeds(seeds)
        else:
            pipeline.run()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    @staticmethod
    def run_probe_hybrid_signal(
        project_root: Path,
        dataset: str,
        nrows: int | None,
        window: int,
        model_config: ModelConfig,
        seed: int,
        checkpoint: Path | None,
    ) -> int:
        """Execute the LSTM-AE-vs-LR correlation probe."""
        config = DatasetConfig(project_root=project_root, dataset_name=dataset, row_limit=nrows)
        sequence_config = SequenceConfig(window_length=window)
        HybridSignalProbe(
            dataset_config=config,
            sequence_config=sequence_config,
            model_config=model_config,
            split_config=SplitConfig(),
            checkpoint_path=checkpoint,
            seed=seed,
        ).run()
        return 0

    @staticmethod
    def run_eval_hybrid(
        project_root: Path,
        dataset: str,
        nrows: int | None,
        window: int,
        model_config: ModelConfig,
        seeds: list[int],
        n_bootstrap: int,
        bootstrap_mode: str = HybridClassifierConfig.bootstrap_mode,
    ) -> int:
        """Execute the P3.5 hybrid classifier evaluation on already-trained checkpoints."""
        config = DatasetConfig(project_root=project_root, dataset_name=dataset, row_limit=nrows)
        HybridClassifierEvaluator(
            dataset_config=config,
            sequence_config=SequenceConfig(window_length=window),
            model_config=model_config,
            config=HybridClassifierConfig(seeds=tuple(seeds), n_bootstrap=n_bootstrap, bootstrap_mode=bootstrap_mode),
        ).run()
        return 0

    @staticmethod
    def run_make_figures(
        project_root: Path, dataset: str, window: int, architecture: str, output_dir: Path | None
    ) -> int:
        """Draw the paper figures from the existing reports (no training, no data loading)."""
        config = DatasetConfig(project_root=project_root, dataset_name=dataset)
        written = FigureBuilder(config, output_dir).build_all(window=window, architecture=architecture)
        print(f"{len(written)} figure(s) written")
        return 0

    @staticmethod
    def run_audit_pipeline(
        project_root: Path,
        dataset: str,
        nrows: int | None,
        windows: str,
        sample_receivers: int,
        seed: int,
    ) -> int:
        """Execute the P0 pipeline audit; the exit code is 1 when any hard check fails."""
        config = DatasetConfig(project_root=project_root, dataset_name=dataset, row_limit=nrows)
        result = PipelineAudit(
            dataset_config=config,
            config=PipelineAuditConfig(
                windows=tuple(int(window) for window in windows.split(",")),
                sample_receivers=sample_receivers,
                seed=seed,
            ),
        ).run()
        return 0 if result["passed"] else 1

    @staticmethod
    def run_eval_standalone(
        project_root: Path,
        dataset: str,
        nrows: int | None,
        windows: str,
        architectures: str,
        seeds: str,
        hidden_dim: int,
        bottleneck_dim: int | None,
        split: str,
        bootstrap: int,
    ) -> int:
        """Score already-trained LSTM-AE checkpoints on a grid of windows and architectures."""
        architecture_list = tuple(name.strip() for name in architectures.split(","))
        unknown = [name for name in architecture_list if name not in STANDALONE_ARCHITECTURES]
        if unknown:
            raise ValueError(f"Unknown architecture(s): {unknown}. Choose from: {list(STANDALONE_ARCHITECTURES)}")
        config = DatasetConfig(project_root=project_root, dataset_name=dataset, row_limit=nrows)
        StandaloneEvaluator(
            dataset_config=config,
            windows=tuple(int(window) for window in windows.split(",")),
            architectures=architecture_list,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            config=StandaloneConfig(
                seeds=tuple(int(seed) for seed in seeds.split(",")), split=split, n_bootstrap=bootstrap
            ),
        ).run()
        return 0

    @staticmethod
    def run_select_score_strategy(
        project_root: Path,
        dataset: str,
        nrows: int | None,
        windows: str,
        architectures: str,
        seeds: str,
        hidden_dim: int,
        bottleneck_dim: int | None,
        force: bool,
    ) -> int:
        """Freeze the mean-vs-last reconstruction-score strategy per configuration from validation AP."""
        architecture_list = tuple(name.strip() for name in architectures.split(","))
        unknown = [name for name in architecture_list if name not in STANDALONE_ARCHITECTURES]
        if unknown:
            raise ValueError(f"Unknown architecture(s): {unknown}. Choose from: {list(STANDALONE_ARCHITECTURES)}")
        config = DatasetConfig(project_root=project_root, dataset_name=dataset, row_limit=nrows)
        ScoreStrategySelector(
            dataset_config=config,
            windows=tuple(int(window) for window in windows.split(",")),
            architectures=architecture_list,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            config=ScoreStrategyConfig(seeds=tuple(int(seed) for seed in seeds.split(",")), force=force),
        ).run()
        return 0

    @staticmethod
    def run_diagnose_scores(
        project_root: Path,
        dataset: str,
        nrows: int | None,
        window: int,
        model_config: ModelConfig,
        seeds: list[int],
        split: str,
    ) -> int:
        """Execute the P2.0/P2.1 reconstruction-score diagnostics on already-trained checkpoints."""
        config = DatasetConfig(project_root=project_root, dataset_name=dataset, row_limit=nrows)
        ScoreDiagnostics(
            dataset_config=config,
            sequence_config=SequenceConfig(window_length=window),
            model_config=model_config,
            config=ScoreDiagnosticsConfig(seeds=tuple(seeds), split=split),
        ).run()
        return 0

    @staticmethod
    def run_eval_unsupervised(
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

    @staticmethod
    def run_train_all_configs(
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
                # Each training run writes into its own version_N_* folder, so search all of them.
                if PaySimModelTrainingPipeline.find_checkpoint(project_root / "artifacts", checkpoint_name) is None:
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

            print(
                f"[{current}/{total_configs}] Training: Window={window}, Architecture={arch_name}, Seeds={seed_list_str}"  # noqa: E501
            )
            print()

            cmd = (
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "lstm_autoencoder",
                    "train-model",
                    "--dataset",
                    dataset,
                    "--window",
                    str(window),
                ]
                + arch_flags
                + ["--seeds", seed_list_str]
            )

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

    @staticmethod
    def run_eval_hybrid_all_configs(
        project_root: Path,
        dataset: str,
        nrows: int | None,
        windows: str,
        seeds: str,
        architectures: str,
        force: bool,
        dry_run: bool,
        bootstrap: int,
        bootstrap_mode: str = HybridClassifierConfig.bootstrap_mode,
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
        print(f"Bootstrap samples: {bootstrap} (mode={bootstrap_mode})")
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
                f"[{current}/{total_configs}] Evaluating: Window={window}, Architecture={arch_name}, Seeds={seed_list_str}"  # noqa: E501
            )
            print()

            cmd = (
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "lstm_autoencoder",
                    "eval-hybrid",
                    "--dataset",
                    dataset,
                    "--window",
                    str(window),
                ]
                + arch_flags
                + [
                    "--seeds",
                    seed_list_str,
                    "--bootstrap",
                    str(bootstrap),
                    "--bootstrap-mode",
                    bootstrap_mode,
                ]
            )

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

    @staticmethod
    def run_pipeline(
        project_root: Path,
        dataset: str,
        nrows: int | None,
        windows: str,
        seeds: str,
        architectures: str,
        bootstrap: int,
        bootstrap_mode: str = HybridClassifierConfig.bootstrap_mode,
    ) -> int:
        """Run end-to-end pipeline: hybrid evaluation, unsupervised evaluation, summarize."""
        import subprocess

        windows_list = [int(w.strip()) for w in windows.split(",")]

        print("Running end-to-end pipeline")
        print(f"Dataset: {dataset}")
        print(f"Windows: {windows_list}")
        print(f"Seeds: {seeds}")
        print(f"Architectures: {architectures}")
        print(f"Bootstrap: {bootstrap} (mode={bootstrap_mode})")
        print()

        # Step 1: Evaluate Hybrid Classifier for all configurations
        for window in windows_list:
            print(f"Evaluating hybrid for window {window}...")
            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "lstm_autoencoder",
                "eval-hybrid-all-configs",
                "--dataset",
                dataset,
                "--windows",
                str(window),
                "--seeds",
                seeds,
                "--architectures",
                architectures,
                "--bootstrap",
                str(bootstrap),
                "--bootstrap-mode",
                bootstrap_mode,
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
            "uv",
            "run",
            "python",
            "-m",
            "lstm_autoencoder",
            "eval-unsupervised",
            "--dataset",
            dataset,
            "--seeds",
            seeds,
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
            "uv",
            "run",
            "python",
            "-m",
            "lstm_autoencoder",
            "summarize-results",
            "--dataset",
            dataset,
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("Summarization failed")
            return 1

        print("Pipeline completed successfully!")
        return 0
