"""RQ D1 unsupervised baselines: Isolation Forest and a dense autoencoder, same split as everything else."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.evaluation.metrics import RankingMetrics
from lstm_autoencoder.evaluation.paysim_synthetic_financial.baseline import (
    HISTORY_STRATA_LABELS,
    BaselineEvaluator,
    FullFeatureData,
)
from lstm_autoencoder.model.config import DenseAutoencoderConfig, SequenceConfig
from lstm_autoencoder.model.dense_autoencoder import DenseAutoencoderTrainer
from lstm_autoencoder.reporting.reporter import ReportWritter

ISOLATION_FOREST_ALL = "isolation_forest_all"
ISOLATION_FOREST_SEQ = "isolation_forest_seq5"
DENSE_AE_SEQ = "dense_ae_seq5"


@dataclass(frozen=True)
class UnsupervisedBaselineConfig:
    """Settings for the unsupervised baselines."""

    seeds: tuple[int, ...] = (42, 123, 2026)
    n_estimators: int = 200
    max_samples: int = 256
    # `account_code` is a factorize id of the receiver, not a behavioural feature;
    # unsupervised models should not consume an arbitrary identifier.
    drop_columns: tuple[str, ...] = ("account_code",)
    fpr_targets: tuple[float, ...] = (0.01, 0.001)


class UnsupervisedBaselineEvaluator:
    """Fit Isolation Forest and a dense autoencoder on train-normal rows, score the test split.

    Models (higher score = more anomalous, all scored on the same test rows as
    the LR / hybrid experiments, via the shared receiver-stratified split):
      isolation_forest_all   every encoded feature except `drop_columns`
      isolation_forest_seq5  the 5 per-transaction features the LSTM-AE sees
      dense_ae_seq5          feed-forward AE on the same 5 features, standardized
                             on train-normal rows (same recipe as FeatureScaler)

    The seq5 models answer RQ D1 / D2 directly: same inputs as the LSTM-AE, no
    recurrence, no attention. Note the dense AE selects its epoch by validation
    reconstruction loss on normal rows, while the LSTM-AE trainer selects by
    validation AUC; this difference is reported, not hidden.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        sequence_config: SequenceConfig | None = None,
        dense_config: DenseAutoencoderConfig | None = None,
        config: UnsupervisedBaselineConfig | None = None,
        reporter: ReportWritter | None = None,
    ) -> None:
        """Bind the evaluator to its configuration."""
        self._dataset_config = dataset_config
        self._sequence_config = sequence_config or SequenceConfig()
        self._dense_config = dense_config or DenseAutoencoderConfig()
        self._config = config or UnsupervisedBaselineConfig()
        self._reporter = reporter or ReportWritter(dataset_config)
        self._reporter.set_path_info_from_module()

    def run(self) -> dict[str, object]:
        """Run every baseline for every seed, report, and return the across-seed aggregate."""
        data = BaselineEvaluator(dataset_config=self._dataset_config).prepare_full_features()
        sequence_columns = list(self._sequence_config.feature_columns)
        missing = [c for c in sequence_columns if c not in data.features.columns]
        if missing:
            raise ValueError(f"Sequence features missing from the baseline feature frame: {missing}")
        all_columns = [c for c in data.features.columns if c not in self._config.drop_columns]

        matrices = {
            ISOLATION_FOREST_ALL: data.features[all_columns].to_numpy(dtype="float64"),
            ISOLATION_FOREST_SEQ: data.features[sequence_columns].to_numpy(dtype="float64"),
        }
        matrices[DENSE_AE_SEQ] = matrices[ISOLATION_FOREST_SEQ]
        y = {name: data.labels[index] for name, index in data.splits.items()}
        buckets_test = data.buckets[data.splits["test"]]

        self._reporter.display_section(
            f"Unsupervised Baselines (seeds={list(self._config.seeds)}, "
            f"dense latent={self._dense_config.latent_dim}, dropped={list(self._config.drop_columns)})"
        )
        self._reporter.display_text(
            f"train normal {int((y['train'] == 0).sum()):,} | val {len(y['val']):,} | "
            f"test {len(y['test']):,} ({int(y['test'].sum()):,} fraud) | "
            f"IF features: all={len(all_columns)}, seq5={len(sequence_columns)}"
        )

        per_seed = [self._run_seed(seed, data, matrices, y, buckets_test) for seed in self._config.seeds]
        aggregate = self._aggregate(per_seed)
        self._report(aggregate)
        self._reporter.update_summary(
            {
                "unsupervised_baselines": {
                    "seeds": list(self._config.seeds),
                    "dropped_columns": list(self._config.drop_columns),
                    "dense_latent_dim": self._dense_config.latent_dim,
                    "per_seed": per_seed,
                    "aggregate": aggregate,
                }
            }
        )
        self._reporter.save_reports(
            text_filename="unsupervised_baselines_report.txt",
            json_filename="unsupervised_baselines_report.json",
        )
        return aggregate

    def _run_seed(
        self,
        seed: int,
        data: FullFeatureData,
        matrices: dict[str, np.ndarray],
        y: dict[str, np.ndarray],
        buckets_test: np.ndarray,
    ) -> dict[str, object]:
        """Fit and score all three baselines for one seed."""
        train, val, test = (data.splits[name] for name in ("train", "val", "test"))
        normal_train = train[y["train"] == 0]
        normal_val = val[y["val"] == 0]
        scores: dict[str, np.ndarray] = {}

        for name in (ISOLATION_FOREST_ALL, ISOLATION_FOREST_SEQ):
            forest = IsolationForest(
                n_estimators=self._config.n_estimators,
                max_samples=self._config.max_samples,
                random_state=seed,
                n_jobs=-1,
            )
            forest.fit(matrices[name][normal_train])
            scores[name] = -forest.score_samples(matrices[name][test])  # higher = more anomalous

        scaler = StandardScaler().fit(matrices[DENSE_AE_SEQ][normal_train])
        trainer = DenseAutoencoderTrainer(self._dense_config)
        model, train_result = trainer.fit(
            scaler.transform(matrices[DENSE_AE_SEQ][normal_train]),
            scaler.transform(matrices[DENSE_AE_SEQ][normal_val]),
            seed,
        )
        scores[DENSE_AE_SEQ] = trainer.predict_errors(model, scaler.transform(matrices[DENSE_AE_SEQ][test]))

        labels = y["test"]
        return {
            "seed": seed,
            "models": {name: self._metrics(values, labels) for name, values in scores.items()},
            "by_bucket": {name: self._by_bucket(values, labels, buckets_test) for name, values in scores.items()},
            "dense_ae_best_epoch": train_result.best_epoch,
            "dense_ae_best_val_loss": train_result.best_val_loss,
        }

    def _metrics(self, scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """AUC, AP and recall at the configured FPR targets."""
        out = {
            "auc": RankingMetrics.roc_auc(scores, labels),
            "ap": RankingMetrics.average_precision(scores, labels),
        }
        for fpr in self._config.fpr_targets:
            out[f"recall_at_fpr_{fpr:g}"] = RankingMetrics.recall_at_fpr(scores, labels, fpr)
        return out

    @staticmethod
    def _by_bucket(
        scores: np.ndarray, labels: np.ndarray, buckets: np.ndarray
    ) -> dict[str, dict[str, float]]:
        """Test AUC/AP per receiver history bucket."""
        rows: dict[str, dict[str, float]] = {}
        for label in HISTORY_STRATA_LABELS:
            mask = buckets == label
            if not mask.any():
                continue
            rows[label] = {
                "n": int(mask.sum()),
                "n_fraud": int(labels[mask].sum()),
                "auc": RankingMetrics.roc_auc(scores[mask], labels[mask]),
                "ap": RankingMetrics.average_precision(scores[mask], labels[mask]),
            }
        return rows

    @staticmethod
    def _mean_std(values: list[float]) -> tuple[float, float]:
        """Mean and std (ddof=1 when more than one value)."""
        array = np.asarray(values, dtype="float64")
        return float(array.mean()), float(array.std(ddof=1 if len(array) > 1 else 0))

    def _aggregate(self, per_seed: list[dict[str, object]]) -> dict[str, object]:
        """Mean/std across seeds for every metric, plus mean AUC per bucket."""
        models: dict[str, dict[str, float]] = {}
        by_bucket: dict[str, dict[str, float]] = {}
        for name, metrics in per_seed[0]["models"].items():  # type: ignore[union-attr]
            row: dict[str, float] = {}
            for metric in metrics:
                mean, std = self._mean_std([run["models"][name][metric] for run in per_seed])  # type: ignore[index]
                row[f"{metric}_mean"], row[f"{metric}_std"] = mean, std
            models[name] = row
            by_bucket[name] = {
                bucket: float(np.mean([run["by_bucket"][name][bucket]["auc"] for run in per_seed]))  # type: ignore[index]
                for bucket in per_seed[0]["by_bucket"][name]  # type: ignore[index]
            }
        epochs = [run["dense_ae_best_epoch"] for run in per_seed]
        return {"n_seeds": len(per_seed), "models": models, "by_bucket_auc": by_bucket, "dense_ae_best_epochs": epochs}

    def _report(self, aggregate: dict[str, object]) -> None:
        """Print the across-seed tables."""
        self._reporter.display_section("Test metrics across seeds (mean / std)")
        self._reporter.display_table(pd.DataFrame(aggregate["models"]).T.round(5))
        self._reporter.display_section("Test AUC by history bucket (mean over seeds)")
        self._reporter.display_table(pd.DataFrame(aggregate["by_bucket_auc"]).T.round(4))
        self._reporter.display_text(f"Dense AE best epochs per seed: {aggregate['dense_ae_best_epochs']}")
        self._reporter.display_text(
            "Reference for the same test rows: `full` LR (ablation report) and the LSTM-AE standalone "
            "results (train-model reports). Compare overall AUC/AP and, above all, the per-bucket AUC, "
            "since between-bucket separation dominates the aggregate numbers."
        )
