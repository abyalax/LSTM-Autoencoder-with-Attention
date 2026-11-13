"""Command CLI for the research project"""

from __future__ import annotations

from collections.abc import Sequence

from lstm_autoencoder.cli.parser import ParserArgs
from lstm_autoencoder.cli.runner import RunnerCLI
from lstm_autoencoder.model.config import ModelConfig, TrainConfig


def _build_model_config(
    bottleneck: bool,
    plain: bool,
    attention_pooling: bool,
    hidden_dim: int,
    bottleneck_dim: int | None,
) -> ModelConfig:
    """Resolve the architecture and capacity flags into one ModelConfig (default architecture: attention)."""
    if sum([bottleneck, plain, attention_pooling]) > 1:
        raise ValueError(
            "--bottleneck, --plain, and --attention-pooling are mutually exclusive. "
            "Choose one architecture or neither (default: attention)."
        )
    return ModelConfig(
        hidden_dim=hidden_dim,
        bottleneck=bottleneck,
        attention=not plain and not bottleneck and not attention_pooling,
        attention_pooling=attention_pooling,
        bottleneck_dim=bottleneck_dim,
    )


class CommandCLI:
    """Command dispatcher for CLI."""

    @staticmethod
    def execute(argv: Sequence[str] | None = None) -> int:
        """Execute the selected CLI command."""
        parser = ParserArgs.build_parser()
        arguments = parser.parse_args(argv)

        project_root = arguments.project_root.resolve()

        commands = {
            "inspect-data": lambda: RunnerCLI.run_inspect_data(project_root, arguments.dataset, arguments.nrows),
            "probe-signal": lambda: RunnerCLI.run_probe_signal(
                project_root,
                arguments.dataset,
                arguments.nrows,
                arguments.max_train_rows,
            ),
            "ablation": lambda: RunnerCLI.run_ablation(
                project_root, arguments.dataset, arguments.nrows, arguments.no_account_code
            ),
            "train-model": lambda: RunnerCLI.run_train_model(
                project_root,
                arguments.dataset,
                arguments.nrows,
                arguments.window,
                _build_model_config(
                    arguments.bottleneck,
                    arguments.plain,
                    arguments.attention_pooling,
                    arguments.hidden_dim,
                    arguments.bottleneck_dim,
                ),
                TrainConfig(
                    max_epochs=arguments.max_epochs,
                    patience=arguments.patience,
                    selection_metric=arguments.selection_metric,
                    min_delta=arguments.min_delta,
                ),
                [int(s) for s in arguments.seeds.split(",")] if arguments.seeds else None,
            ),
            "eval-hybrid": lambda: RunnerCLI.run_eval_hybrid(
                project_root,
                arguments.dataset,
                arguments.nrows,
                arguments.window,
                _build_model_config(
                    arguments.bottleneck,
                    arguments.plain,
                    arguments.attention_pooling,
                    arguments.hidden_dim,
                    arguments.bottleneck_dim,
                ),
                [int(seed) for seed in arguments.seeds.split(",")],
                arguments.bootstrap,
                arguments.bootstrap_mode,
            ),
            "eval-banksim": lambda: RunnerCLI.run_eval_banksim(
                project_root,
                arguments.dataset,
                arguments.nrows,
                arguments.window,
                _build_model_config(
                    arguments.bottleneck,
                    arguments.plain,
                    arguments.attention_pooling,
                    arguments.hidden_dim,
                    arguments.bottleneck_dim,
                ),
                [int(seed) for seed in arguments.seeds.split(",")],
                arguments.bootstrap,
            ),
            "diagnose-scores": lambda: RunnerCLI.run_diagnose_scores(
                project_root,
                arguments.dataset,
                arguments.nrows,
                arguments.window,
                _build_model_config(
                    arguments.bottleneck,
                    arguments.plain,
                    arguments.attention_pooling,
                    arguments.hidden_dim,
                    arguments.bottleneck_dim,
                ),
                [int(seed) for seed in arguments.seeds.split(",")],
                arguments.split,
            ),
            "select-score-strategy": lambda: RunnerCLI.run_select_score_strategy(
                project_root,
                arguments.dataset,
                arguments.nrows,
                arguments.windows,
                arguments.architectures,
                arguments.seeds,
                arguments.hidden_dim,
                arguments.bottleneck_dim,
                arguments.force,
            ),
            "make-figures": lambda: RunnerCLI.run_make_figures(
                project_root,
                arguments.dataset,
                arguments.window,
                arguments.architecture,
                arguments.output_dir,
            ),
            "audit-pipeline": lambda: RunnerCLI.run_audit_pipeline(
                project_root,
                arguments.dataset,
                arguments.nrows,
                arguments.windows,
                arguments.sample_receivers,
                arguments.seed,
            ),
            "eval-standalone": lambda: RunnerCLI.run_eval_standalone(
                project_root,
                arguments.dataset,
                arguments.nrows,
                arguments.windows,
                arguments.architectures,
                arguments.seeds,
                arguments.hidden_dim,
                arguments.bottleneck_dim,
                arguments.split,
                arguments.bootstrap,
            ),
            "eval-unsupervised": lambda: RunnerCLI.run_eval_unsupervised(
                project_root,
                arguments.dataset,
                arguments.nrows,
                [int(seed) for seed in arguments.seeds.split(",")],
                arguments.keep_account_code,
                arguments.latent_dim,
                arguments.hidden_dim,
            ),
            "summarize-results": lambda: RunnerCLI.run_summarize_results(
                project_root, arguments.dataset, arguments.out, arguments.legacy_since
            ),
            "probe-hybrid-signal": lambda: RunnerCLI.run_probe_hybrid_signal(
                project_root,
                arguments.dataset,
                arguments.nrows,
                arguments.window,
                _build_model_config(
                    arguments.bottleneck,
                    arguments.plain,
                    arguments.attention_pooling,
                    arguments.hidden_dim,
                    arguments.bottleneck_dim,
                ),
                arguments.seed,
                arguments.checkpoint,
            ),
            "train-all-configs": lambda: RunnerCLI.run_train_all_configs(
                project_root,
                arguments.dataset,
                arguments.nrows,
                arguments.windows,
                arguments.seeds,
                arguments.architectures,
                arguments.force,
                arguments.dry_run,
            ),
            "eval-hybrid-all-configs": lambda: RunnerCLI.run_eval_hybrid_all_configs(
                project_root,
                arguments.dataset,
                arguments.nrows,
                arguments.windows,
                arguments.seeds,
                arguments.architectures,
                arguments.force,
                arguments.dry_run,
                arguments.bootstrap,
                arguments.bootstrap_mode,
            ),
            "run-pipeline": lambda: RunnerCLI.run_pipeline(
                project_root,
                arguments.dataset,
                arguments.nrows,
                arguments.windows,
                arguments.seeds,
                arguments.architectures,
                arguments.bootstrap,
                arguments.bootstrap_mode,
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
