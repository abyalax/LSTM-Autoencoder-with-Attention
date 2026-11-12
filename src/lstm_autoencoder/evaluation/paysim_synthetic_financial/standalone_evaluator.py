"""Standalone LSTM-AE evaluation from existing checkpoints: AUC, AP, Recall@FPR and receiver-cluster CIs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.evaluation.bootstrap import WeightedRankingBootstrap
from lstm_autoencoder.evaluation.config import SplitConfig
from lstm_autoencoder.evaluation.metrics import RankingMetrics
from lstm_autoencoder.evaluation.paysim_synthetic_financial.score_strategy_registry import FrozenScoreStrategy
from lstm_autoencoder.model.config import (
    DEFAULT_HIDDEN_DIM,
    SCORE_STRATEGIES,
    ModelConfig,
    SequenceConfig,
    TrainConfig,
)
from lstm_autoencoder.model.trainer import Trainer
from lstm_autoencoder.pipeline.paysim.paysim_pipeline import PaySimModelTrainingPipeline, SplitScoringData
from lstm_autoencoder.reporting.reporter import ReportWritter

ARCHITECTURES: tuple[str, ...] = ("attention", "bottleneck", "plain", "attention_pooling")
BUCKETS: tuple[str, ...] = ("1", "2-4", "5-9", ">=10")
SPLITS: tuple[str, ...] = ("val", "test")


@dataclass(frozen=True)
class StandaloneConfig:
    """Settings of the standalone evaluation.

    `n_bootstrap` receiver-cluster resamples give a 95% CI per run for every metric
    (0 skips it). The split defaults to `test`: these are the numbers that get reported.
    """

    seeds: tuple[int, ...] = (42, 123, 2026)
    split: str = "test"
    fpr_targets: tuple[float, ...] = (0.01, 0.001)
    n_bootstrap: int = 2000
    random_seed: int = 42
    # None = use the strategy frozen on validation (`select-score-strategy`), which is
    # REQUIRED for the test split. An explicit value is only allowed on validation.
    score_strategy: str | None = None

    def __post_init__(self) -> None:
        """Validate the split name and the strategy override."""
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {self.split!r}.")
        if self.score_strategy is not None:
            if self.score_strategy not in SCORE_STRATEGIES:
                raise ValueError(f"score_strategy must be one of {SCORE_STRATEGIES}, got {self.score_strategy!r}.")
            if self.split == "test":
                raise ValueError("The test split must use the frozen strategy; do not override it by hand.")


def model_config_for(
    architecture: str, hidden_dim: int = DEFAULT_HIDDEN_DIM, bottleneck_dim: int | None = None
) -> ModelConfig:
    """Map a CLI architecture name ('attention', 'bottleneck', 'plain', 'attention_pooling') to a ModelConfig."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}. Choose from: {ARCHITECTURES}")
    return ModelConfig(
        hidden_dim=hidden_dim,
        attention=architecture == "attention",
        bottleneck=architecture == "bottleneck",
        attention_pooling=architecture == "attention_pooling",
        bottleneck_dim=bottleneck_dim,
    )


class StandaloneEvaluator:
    """Score already-trained LSTM-AE checkpoints on a held-out split, for every window x architecture.

    No retraining: checkpoints are located with `PaySimModelTrainingPipeline.checkpoint_path`
    (newest `version_N_*` folder holding that exact name). For each run it reports AUC, average
    precision, recall at the configured FPRs, the same metrics inside each history bucket (with
    the bucket fraud rate), a receiver-cluster bootstrap CI, and the model's parameter count.
    The data of one window is prepared once and shared by all architectures.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        windows: tuple[int, ...] = (1, 3, 5, 10),
        architectures: tuple[str, ...] = ARCHITECTURES,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        bottleneck_dim: int | None = None,
        split_config: SplitConfig | None = None,
        config: StandaloneConfig | None = None,
        reporter: ReportWritter | None = None,
    ) -> None:
        """Bind the evaluator to the grid of configurations to score."""
        self._dataset_config = dataset_config
        self._windows = windows
        self._architectures = architectures
        self._hidden_dim = hidden_dim
        self._bottleneck_dim = bottleneck_dim
        self._split_config = split_config or SplitConfig()
        self._config = config or StandaloneConfig()
        self._reporter = reporter or ReportWritter(dataset_config)
        self._reporter.set_path_info_from_module()
        # Fail on an invalid architecture/capacity combination before any data is loaded.
        for architecture in architectures:
            self._model_config(architecture)

    def run(self) -> list[dict[str, object]]:
        """Score every configuration, write the report, and return one record per (architecture, window)."""
        self._reporter.display_section(
            f"StandAlone LSTM-AE Evaluation (split={self._config.split}, seeds={list(self._config.seeds)}, "
            f"bootstrap={self._config.n_bootstrap})"
        )
        records: list[dict[str, object]] = []
        for window in self._windows:
            sequence_config = SequenceConfig(window_length=window)
            data = self._load_split(sequence_config)
            for architecture in self._architectures:
                record = self._evaluate_config(architecture, sequence_config, data)
                records.append(record)
                self._report_config(record)

        self._reporter.display_section("Summary (mean +/- std across seeds)")
        self._reporter.display_table(self._summary_table(records))
        self._reporter.update_summary(
            {
                "run_config": {
                    "split": self._config.split,
                    "seeds": list(self._config.seeds),
                    "fpr_targets": list(self._config.fpr_targets),
                    "n_bootstrap": self._config.n_bootstrap,
                    "score_strategy": self._config.score_strategy or "frozen (per configuration, see records)",
                    "bootstrap_mode": "receiver" if self._config.n_bootstrap > 0 else "none",
                    "hidden_dim": self._hidden_dim,
                    "bottleneck_dim": self._bottleneck_dim,
                },
                "standalone": {"configs": records},
            }
        )
        self._reporter.save_reports(text_filename="standalone_report.txt", json_filename="standalone_report.json")
        return records

    def _model_config(self, architecture: str) -> ModelConfig:
        """ModelConfig of one architecture under the shared capacity settings."""
        return model_config_for(architecture, self._hidden_dim, self._bottleneck_dim)

    def _load_split(self, sequence_config: SequenceConfig) -> SplitScoringData:
        """Prepare the chosen split of one window once (the data does not depend on the architecture)."""
        pipeline = PaySimModelTrainingPipeline(
            dataset_config=self._dataset_config,
            sequence_config=sequence_config,
            model_config=ModelConfig(),
            split_config=self._split_config,
        )
        data = pipeline.prepared_eval_splits()[self._config.split]
        if data.history_bucket is None or data.receiver is None:
            raise ValueError("The prepared split lacks history buckets or receivers.")
        return data

    def _evaluate_config(
        self, architecture: str, sequence_config: SequenceConfig, data: SplitScoringData
    ) -> dict[str, object]:
        """Score all seeds of one (architecture, window) configuration."""
        model_config = self._model_config(architecture)
        pipeline = PaySimModelTrainingPipeline(
            dataset_config=self._dataset_config,
            sequence_config=sequence_config,
            model_config=model_config,
            split_config=self._split_config,
        )
        runs: list[dict[str, object]] = []
        missing: list[int] = []
        parameter_count: int | None = None
        existing = [path for path in (pipeline.checkpoint_path(seed) for seed in self._config.seeds) if path.exists()]
        strategy = self._resolve_strategy(model_config.architecture_tag, sequence_config.window_length, existing)
        for seed in self._config.seeds:
            checkpoint = pipeline.checkpoint_path(seed)
            if not checkpoint.exists():
                missing.append(seed)
                continue
            trainer = Trainer(
                model_config, TrainConfig(seed=seed, fpr_targets=self._config.fpr_targets), checkpoint.parent
            )
            model = trainer.build_eval_model(checkpoint)
            parameter_count = model.parameter_count
            errors = trainer.predict_errors(model, data.dataset, strategy)
            runs.append(self._score_run(errors, data, seed, str(checkpoint)))
        return {
            "architecture": model_config.architecture_tag,
            "window_length": sequence_config.window_length,
            "split": self._config.split,
            "hidden_dim": model_config.hidden_dim,
            "bottleneck_dim": model_config.bottleneck_dim,
            "num_layers": model_config.num_layers,
            "score_strategy": strategy,
            "parameter_count": parameter_count,
            "seeds_found": [int(run["seed"]) for run in runs],  # type: ignore[arg-type]
            "seeds_missing": missing,
            "runs": runs,
            "aggregate": self._aggregate(runs),
        }

    def _resolve_strategy(self, architecture_tag: str, window: int, checkpoints: list) -> str:
        """Frozen strategy (default) or the explicit validation-only override."""
        if self._config.score_strategy is not None:
            return self._config.score_strategy
        frozen = FrozenScoreStrategy.load(self._dataset_config.artifacts_dir)
        return frozen.strategy_for(architecture_tag, window, checkpoints)

    def _score_run(self, errors: np.ndarray, data: SplitScoringData, seed: int, checkpoint: str) -> dict[str, object]:
        """Metrics, per-bucket metrics and bootstrap CIs of one checkpoint."""
        labels = np.asarray(data.labels).astype(int)
        buckets = np.asarray(data.history_bucket)
        metrics = self._metrics(errors, labels)
        by_bucket: dict[str, dict[str, float]] = {}
        for bucket in BUCKETS:
            mask = buckets == bucket
            if not mask.any():
                continue
            n_fraud = int(labels[mask].sum())
            by_bucket[bucket] = {
                "n": int(mask.sum()),
                "n_fraud": n_fraud,
                "fraud_rate": n_fraud / int(mask.sum()),
                "auc": float(RankingMetrics.roc_auc(errors[mask], labels[mask])),
                "ap": float(RankingMetrics.average_precision(errors[mask], labels[mask])),
            }
        run: dict[str, object] = {
            "seed": seed,
            "checkpoint": checkpoint,
            "n": int(len(labels)),
            "n_fraud": int(labels.sum()),
            "prevalence": float(labels.mean()),
            "metrics": metrics,
            "by_bucket": by_bucket,
        }
        if self._config.n_bootstrap > 0:
            run["bootstrap_ci"] = self._bootstrap_ci(errors, labels, np.asarray(data.receiver), seed)
        return run

    def _metrics(self, errors: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """AUC, AP and recall at each configured FPR."""
        out = {
            "auc": float(RankingMetrics.roc_auc(errors, labels)),
            "ap": float(RankingMetrics.average_precision(errors, labels)),
        }
        for fpr in self._config.fpr_targets:
            out[f"recall_at_fpr_{fpr:g}"] = float(RankingMetrics.recall_at_fpr(errors, labels, fpr))
        return out

    def _bootstrap_ci(
        self, errors: np.ndarray, labels: np.ndarray, receivers: np.ndarray, seed: int
    ) -> dict[str, float]:
        """95% percentile CI of every metric from `n_bootstrap` receiver-cluster resamples."""
        rng = np.random.default_rng(self._config.random_seed + 3_000 + seed)
        engine = WeightedRankingBootstrap(labels, {"ae": errors}, self._config.fpr_targets)
        _, inverse = np.unique(receivers, return_inverse=True)
        n_receivers = int(inverse.max()) + 1
        draws: dict[str, list[float]] = {}
        for _ in range(self._config.n_bootstrap):
            for key, value in engine.metrics(engine.cluster_weights(inverse, n_receivers, rng))["ae"].items():
                draws.setdefault(key, []).append(value)
        ci: dict[str, float] = {}
        for key, values in draws.items():
            low, high = np.nanpercentile(values, [2.5, 97.5])
            ci[f"{key}_ci_low"], ci[f"{key}_ci_high"] = float(low), float(high)
        return ci

    @staticmethod
    def _aggregate(runs: list[dict[str, object]]) -> dict[str, float]:
        """Mean and std (ddof=1 when more than one seed) of every metric across seeds."""
        if not runs:
            return {}
        ddof = 1 if len(runs) > 1 else 0
        out: dict[str, float] = {"n_seeds": float(len(runs))}
        for key in runs[0]["metrics"]:  # type: ignore[union-attr]
            values = np.array([run["metrics"][key] for run in runs], dtype="float64")  # type: ignore[index]
            out[f"{key}_mean"] = float(np.nanmean(values))
            out[f"{key}_std"] = float(np.nanstd(values, ddof=ddof)) if len(values) > ddof else 0.0
        return out

    def _report_config(self, record: dict[str, object]) -> None:
        """One console line per configuration."""
        aggregate: dict[str, float] = record["aggregate"]  # type: ignore[assignment]
        head = f"[{record['architecture']} W={record['window_length']}]"
        if not aggregate:
            self._reporter.display_text(f"{head} no checkpoints found (missing seeds {record['seeds_missing']})")
            return
        recalls = " | ".join(
            f"recall@{float(key[len('recall_at_fpr_') : -len('_mean')]) * 100:g}%={aggregate[key]:.4f}"
            for key in aggregate
            if key.startswith("recall_at_fpr_") and key.endswith("_mean")
        )
        missing = f" | MISSING seeds {record['seeds_missing']}" if record["seeds_missing"] else ""
        self._reporter.display_text(
            f"{head} AUC {aggregate['auc_mean']:.4f} +/- {aggregate['auc_std']:.4f} | "
            f"AP {aggregate['ap_mean']:.4f} +/- {aggregate['ap_std']:.4f} | {recalls}{missing}"
        )

    @staticmethod
    def _summary_table(records: list[dict[str, object]]) -> pd.DataFrame:
        """Architecture x window table of the aggregated metrics, as strings for the console grid."""
        rows = []
        for record in records:
            aggregate: dict[str, float] = record["aggregate"]  # type: ignore[assignment]
            if not aggregate:
                continue
            row: dict[str, object] = {"architecture": record["architecture"], "W": record["window_length"]}
            for key in aggregate:
                if key.endswith("_mean"):
                    base = key[: -len("_mean")]
                    row[base] = f"{aggregate[key]:.4f} +/- {aggregate[base + '_std']:.4f}"
            row["params"] = record["parameter_count"]
            rows.append(row)
        return pd.DataFrame(rows)
