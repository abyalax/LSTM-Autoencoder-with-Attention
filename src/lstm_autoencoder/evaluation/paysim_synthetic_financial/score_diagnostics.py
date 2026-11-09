"""P2.0/P2.1 score diagnostics: does LSTM-AE reconstruction error track fraud, or window complexity?"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.evaluation.config import SplitConfig
from lstm_autoencoder.evaluation.metrics import RankingMetrics
from lstm_autoencoder.evaluation.paysim_synthetic_financial.baseline import (
    HISTORY_STRATA_LABELS,
    BaselineEvaluator,
)
from lstm_autoencoder.model.config import ModelConfig, SequenceConfig, TrainConfig
from lstm_autoencoder.model.pipeline import PaySimModelTrainingPipeline, SplitScoringData
from lstm_autoencoder.model.trainer import Trainer
from lstm_autoencoder.reporting.reporter import ReportWritter

SCORE_DEFINITIONS: tuple[str, ...] = ("mean", "last", "max")
GROUPED_SCORES: tuple[str, ...] = ("mean", "last")
SPLITS: tuple[str, ...] = ("val", "test")
COUNT_COLUMNS: tuple[str, ...] = ("n", "n_fraud")


@dataclass(frozen=True)
class ScoreDiagnosticsConfig:
    """Settings for the score diagnostics.

    `split` defaults to `val`: diagnostics inform design decisions (capacity,
    checkpoint selection), and those must not be made by looking at test. Use
    `test` only for the final reported numbers; it also enables the LR-outcome groups.
    """

    seeds: tuple[int, ...] = (42,)
    split: str = "val"
    fpr_targets: tuple[float, ...] = (0.01, 0.001)

    def __post_init__(self) -> None:
        """Validate the split name."""
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {self.split!r}.")


class ScoreDiagnostics:
    """Compare reconstruction-score definitions and test whether error tracks history length.

    Loads already-trained checkpoints (no retraining) and scores one held-out split
    with three definitions of the per-window anomaly score:

      mean  masked mean over the real steps (what training and every report used so far)
      last  error of the current/final transaction only (the one that carries the label)
      max   largest error over the real steps

    It then reports, per seed: AUC/AP/recall@FPR per definition, error distribution
    (count, median, p90, p99) per outcome group, rank correlation of the error with the
    number of real steps and the history bucket, and AE-only AUC/AP INSIDE each history
    bucket. A pooled AUC well below 0.5 next to within-bucket AUCs near 0.5 means the
    error is tracking history length (window complexity), not fraud.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        sequence_config: SequenceConfig | None = None,
        model_config: ModelConfig | None = None,
        split_config: SplitConfig | None = None,
        config: ScoreDiagnosticsConfig | None = None,
        reporter: ReportWritter | None = None,
    ) -> None:
        """Bind the diagnostics to the configuration of the checkpoints to inspect."""
        self._dataset_config = dataset_config
        self._sequence_config = sequence_config or SequenceConfig()
        self._model_config = model_config or ModelConfig()
        self._split_config = split_config or SplitConfig()
        self._config = config or ScoreDiagnosticsConfig()
        self._reporter = reporter or ReportWritter(dataset_config)
        self._reporter.set_path_info_from_module()

    def run(self) -> dict[str, object]:
        """Score every seed's checkpoint on the chosen split, report, and return the aggregate."""
        pipeline = PaySimModelTrainingPipeline(
            dataset_config=self._dataset_config,
            sequence_config=self._sequence_config,
            model_config=self._model_config,
            split_config=self._split_config,
        )
        data = pipeline.prepared_eval_splits()[self._config.split]
        if data.history_bucket is None:
            raise ValueError("History buckets are missing from the prepared split.")
        valid_steps = data.dataset.valid_steps()
        lr_scores = self._lr_scores(data) if self._config.split == "test" else None

        self._reporter.display_section(
            f"Score diagnostics (split={self._config.split}, W={self._sequence_config.window_length}, "
            f"architecture={self._model_config.architecture_tag})"
        )
        self._reporter.display_text(
            f"n {len(data.labels):,} | fraud {int(data.labels.sum()):,} | seeds {list(self._config.seeds)}"
        )
        if lr_scores is None:
            self._reporter.display_text("LR-outcome groups skipped (the LR scores are only aligned on the test split).")

        per_seed: list[dict[str, object]] = []
        for seed in self._config.seeds:
            checkpoint = pipeline.checkpoint_path(seed)
            if not checkpoint.exists():
                raise ValueError(
                    f"Checkpoint not found: {checkpoint}. Train it first with `train-model "
                    f"--window {self._sequence_config.window_length} --seeds {seed}` and the same "
                    "architecture, --hidden-dim and --bottleneck-dim flags."
                )
            trainer = Trainer(self._model_config, TrainConfig(seed=seed), checkpoint.parent)
            model = trainer.build_eval_model(checkpoint)
            steps = trainer.predict_step_errors(model, data.dataset)
            scores = self._scores(steps, valid_steps)
            result = self._diagnose_seed(seed, str(checkpoint), scores, data, valid_steps, lr_scores)
            per_seed.append(result)
            self._report_seed(result)

        aggregate = self._aggregate(per_seed)
        self._report_aggregate(aggregate)
        self._record_run_config()
        self._reporter.update_summary(
            {
                "score_diagnostics": {
                    "split": self._config.split,
                    "window_length": self._sequence_config.window_length,
                    "architecture": self._model_config.architecture_tag,
                    "per_seed": per_seed,
                    "aggregate": aggregate,
                }
            }
        )
        self._reporter.save_reports(
            text_filename="score_diagnostics_report.txt",
            json_filename="score_diagnostics_report.json",
        )
        return aggregate

    def _lr_scores(self, data: SplitScoringData) -> np.ndarray:
        """Fit the LR 'full' baseline and return its test scores, row-aligned with the AE windows."""
        test_idx, predicted, labels = BaselineEvaluator(dataset_config=self._dataset_config).fit_full_variant()
        if not np.array_equal(test_idx, data.row_index) or not np.array_equal(labels, data.labels):
            raise ValueError(
                "LR and LSTM-AE test rows differ -- check both use the same --dataset, --nrows, and split seed."
            )
        return predicted

    @staticmethod
    def _scores(steps: np.ndarray, valid_steps: np.ndarray) -> dict[str, np.ndarray]:
        """Turn per-step errors (n, W) into one score per window.

        Windows are left-padded and padded steps carry 0 error, so the last column is
        always the current transaction and the max over all columns equals the max over real steps.
        """
        return {
            "mean": steps.sum(axis=1) / np.maximum(valid_steps, 1),
            "last": steps[:, -1],
            "max": steps.max(axis=1),
        }

    def _diagnose_seed(
        self,
        seed: int,
        checkpoint: str,
        scores: dict[str, np.ndarray],
        data: SplitScoringData,
        valid_steps: np.ndarray,
        lr_scores: np.ndarray | None,
    ) -> dict[str, object]:
        """Compute every diagnostic for one seed's checkpoint."""
        labels = np.asarray(data.labels).astype(int)
        buckets = np.asarray(data.history_bucket)
        return {
            "seed": seed,
            "checkpoint": checkpoint,
            "score_definitions": self._score_definitions(scores, labels),
            "group_stats": {name: self._group_stats(scores[name], labels, lr_scores) for name in GROUPED_SCORES},
            "correlations": self._correlations(scores["mean"], labels, valid_steps, buckets),
            "per_bucket": self._per_bucket(scores, labels, buckets),
        }

    def _score_definitions(self, scores: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, dict[str, float]]:
        """AUC, AP and recall at fixed FPR for each score definition, on identical rows and labels."""
        rows: dict[str, dict[str, float]] = {}
        for name in SCORE_DEFINITIONS:
            row = {
                "auc": RankingMetrics.roc_auc(scores[name], labels),
                "ap": RankingMetrics.average_precision(scores[name], labels),
            }
            for fpr in self._config.fpr_targets:
                row[f"recall@FPR{fpr:.1%}"] = RankingMetrics.recall_at_fpr(scores[name], labels, fpr)
            rows[name] = {key: float(value) for key, value in row.items()}
        return rows

    @staticmethod
    def _group_stats(
        score: np.ndarray, labels: np.ndarray, lr_scores: np.ndarray | None
    ) -> dict[str, dict[str, float]]:
        """Count and robust distribution statistics of the error per outcome group."""
        fraud = labels == 1
        groups = {"normal": labels == 0, "fraud": fraud}
        if lr_scores is not None and fraud.any():
            low = fraud & (lr_scores <= np.median(lr_scores[fraud]))
            groups["fraud_lr_low"] = low  # fraud the LR scored in its lower half: its blind spot
            groups["fraud_lr_high"] = fraud & ~low
        rows: dict[str, dict[str, float]] = {}
        for name, mask in groups.items():
            values = score[mask]
            empty = len(values) == 0
            rows[name] = {
                "n": int(mask.sum()),
                "median": float("nan") if empty else float(np.quantile(values, 0.5)),
                "p90": float("nan") if empty else float(np.quantile(values, 0.9)),
                "p99": float("nan") if empty else float(np.quantile(values, 0.99)),
            }
        return rows

    @classmethod
    def _correlations(
        cls, score: np.ndarray, labels: np.ndarray, valid_steps: np.ndarray, buckets: np.ndarray
    ) -> dict[str, dict[str, float]]:
        """Spearman correlation of the mean error with real-step count and history bucket (0 = shortest)."""
        ordinal = (
            pd.Series(buckets)
            .map({label: index for index, label in enumerate(HISTORY_STRATA_LABELS)})
            .to_numpy(dtype="float64")
        )
        groups = {"all": np.ones(len(labels), dtype=bool), "normal": labels == 0, "fraud": labels == 1}
        return {
            name: {
                "spearman_vs_valid_steps": cls._spearman(score[mask], valid_steps[mask]),
                "spearman_vs_history_bucket": cls._spearman(score[mask], ordinal[mask]),
            }
            for name, mask in groups.items()
        }

    @staticmethod
    def _per_bucket(
        scores: dict[str, np.ndarray], labels: np.ndarray, buckets: np.ndarray
    ) -> dict[str, dict[str, float]]:
        """AE-only AUC/AP inside each history bucket, plus the pooled and fraud-weighted within-bucket AUC."""
        rows: dict[str, dict[str, float]] = {}
        weighted_sum, weight = 0.0, 0
        for label in HISTORY_STRATA_LABELS:
            mask = buckets == label
            n_fraud = int(labels[mask].sum())
            row: dict[str, float] = {
                "n": int(mask.sum()),
                "n_fraud": n_fraud,
                "fraud_rate": float(n_fraud / mask.sum()) if mask.any() else float("nan"),
            }
            for name in GROUPED_SCORES:
                row[f"auc_{name}"] = float(RankingMetrics.roc_auc(scores[name][mask], labels[mask]))
                row[f"ap_{name}"] = float(RankingMetrics.average_precision(scores[name][mask], labels[mask]))
            if np.isfinite(row["auc_mean"]):
                weighted_sum += row["auc_mean"] * n_fraud
                weight += n_fraud
            rows[label] = row
        rows["pooled"] = {
            "n": int(len(labels)),
            "n_fraud": int(labels.sum()),
            "fraud_rate": float(labels.mean()),
            **{f"auc_{name}": float(RankingMetrics.roc_auc(scores[name], labels)) for name in GROUPED_SCORES},
            **{f"ap_{name}": float(RankingMetrics.average_precision(scores[name], labels)) for name in GROUPED_SCORES},
        }
        rows["within_bucket_weighted"] = {"auc_mean": weighted_sum / weight if weight else float("nan")}
        return rows

    @staticmethod
    def _spearman(x: np.ndarray, y: np.ndarray) -> float:
        """Spearman rank correlation; NaN when either side is constant or there are too few rows."""
        if len(x) < 3:
            return float("nan")
        ranked_x = pd.Series(np.asarray(x, dtype="float64")).rank().to_numpy()
        ranked_y = pd.Series(np.asarray(y, dtype="float64")).rank().to_numpy()
        if ranked_x.std() == 0 or ranked_y.std() == 0:
            return float("nan")
        return float(np.corrcoef(ranked_x, ranked_y)[0, 1])

    @staticmethod
    def _aggregate(per_seed: list[dict[str, object]]) -> dict[str, dict[str, float]]:
        """Mean/std across seeds of every score-definition metric."""
        ddof = 1 if len(per_seed) > 1 else 0
        out: dict[str, dict[str, float]] = {}
        for name in SCORE_DEFINITIONS:
            first: dict[str, float] = per_seed[0]["score_definitions"][name]  # type: ignore[index]
            row: dict[str, float] = {}
            for metric in first:
                values = np.array([run["score_definitions"][name][metric] for run in per_seed], dtype="float64")  # type: ignore[index]
                row[f"{metric}_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
                row[f"{metric}_std"] = float(np.nanstd(values, ddof=ddof)) if np.isfinite(values).sum() > ddof else 0.0
            out[name] = row
        return out

    def _record_run_config(self) -> None:
        """Write the resolved configuration of the inspected checkpoints into the report."""
        config = self._model_config
        self._reporter.update_summary(
            {
                "run_config": {
                    "architecture": config.architecture_tag,
                    "window_length": self._sequence_config.window_length,
                    "seeds": list(self._config.seeds),
                    "split": self._config.split,
                    "model_config": {
                        "attention": config.attention,
                        "bottleneck": config.bottleneck,
                        "attention_pooling": config.attention_pooling,
                        "hidden_dim": config.hidden_dim,
                        "bottleneck_dim": config.bottleneck_dim,
                        "num_layers": config.num_layers,
                        "dropout": config.dropout,
                    },
                }
            }
        )

    def _report_seed(self, result: dict[str, object]) -> None:
        """Print the tables of one seed."""
        self._reporter.display_section(f"Seed {result['seed']}")
        self._reporter.display_text("Score definitions (same rows, same labels):")
        self._reporter.display_table(self._table(result["score_definitions"]))  # type: ignore[arg-type]
        for name in GROUPED_SCORES:
            self._reporter.display_text(f"\nError distribution by group ({name} score):")
            self._reporter.display_table(self._table(result["group_stats"][name]))  # type: ignore[index]
        self._reporter.display_text(
            "\nSpearman of the mean error vs real-step count and history bucket "
            "(normal rows only isolates the confound from the label):"
        )
        self._reporter.display_table(self._table(result["correlations"]))  # type: ignore[arg-type]
        self._reporter.display_text(
            "\nAE-only metrics inside each history bucket (pooled = the number reported so far):"
        )
        self._reporter.display_table(self._table(result["per_bucket"]))  # type: ignore[arg-type]

    def _report_aggregate(self, aggregate: dict[str, dict[str, float]]) -> None:
        """Print the across-seed summary of the score definitions."""
        self._reporter.display_section("Across seeds (mean +/- std)")
        rows = {}
        for name, metrics in aggregate.items():
            rows[name] = {
                key[: -len("_mean")]: f"{metrics[key]:.4f} +/- {metrics[key[: -len('_mean')] + '_std']:.4f}"
                for key in metrics
                if key.endswith("_mean")
            }
        self._reporter.display_table(pd.DataFrame(rows).T)

    @classmethod
    def _table(cls, rows: dict[str, dict[str, float]]) -> pd.DataFrame:
        """Rows-by-metrics DataFrame of pre-formatted strings, for the console grid."""
        frame = pd.DataFrame(rows).T
        return frame.apply(
            lambda column: column.map(lambda value: cls._format(value, is_count=column.name in COUNT_COLUMNS))
        )

    @staticmethod
    def _format(value: object, is_count: bool = False) -> str:
        """Compact, comparable cell formatting: counts as integers, small errors in scientific notation."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "nan"
        number = float(value)  # type: ignore[arg-type]
        if is_count:
            return f"{int(number):,}"
        if number != 0 and abs(number) < 1e-3:
            return f"{number:.2e}"
        return f"{number:.4f}"
