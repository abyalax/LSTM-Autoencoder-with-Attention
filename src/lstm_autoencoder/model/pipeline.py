"""End-to-end training pipeline: load -> causal features -> scale -> sequences -> train -> evaluate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.loader import DatasetLoader
from lstm_autoencoder.data.paysim_synthetic_financial.causal_features import (
    HISTORY_LENGTH,
    CausalFeatureBuilder,
)
from lstm_autoencoder.data.paysim_synthetic_financial.schema import PaySimSchema
from lstm_autoencoder.evaluation.config import SplitConfig
from lstm_autoencoder.model.config import ModelConfig, SequenceConfig, TrainConfig
from lstm_autoencoder.model.dataset import TransactionSequenceDataset
from lstm_autoencoder.model.feature_scaler import FeatureScaler
from lstm_autoencoder.model.sequence_builder import SequenceBatch, SequenceBuilder
from lstm_autoencoder.model.trainer import Trainer
from lstm_autoencoder.reporting.reporter import ReportWritter

FRAUD_RATE_BINS: tuple[float, ...] = (0, 0.001, 0.01, 0.1, 1.0)
FRAUD_RATE_LABELS: tuple[str, ...] = ("very_low", "low", "medium", "high")


@dataclass(frozen=True)
class _PreparedData:
    """Train/val/test subsets and labels, built once and reused across seeds."""

    train_subset: TransactionSequenceDataset
    val_subset: TransactionSequenceDataset
    test_subset: TransactionSequenceDataset
    val_labels: np.ndarray
    test_labels: np.ndarray
    test_row_index: np.ndarray
    n_train_normal: int
    n_val: int
    n_test: int
    # Only filled when `_prepare_data(with_train_scoring=True)`: scoring a trained
    # checkpoint on ALL train rows (normal + fraud) needs its own dataset copy.
    train_all_subset: TransactionSequenceDataset | None = None
    train_all_labels: np.ndarray | None = None
    train_row_index: np.ndarray | None = None
    val_row_index: np.ndarray | None = None


@dataclass(frozen=True)
class SplitScoringData:
    """Windows, labels and batch-row positions of one split, for scoring a trained checkpoint."""

    dataset: TransactionSequenceDataset
    labels: np.ndarray
    row_index: np.ndarray


class PaySimModelTrainingPipeline:
    """Orchestrate loading, scaling, sequence building, training, and evaluation."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        sequence_config: SequenceConfig | None = None,
        model_config: ModelConfig | None = None,
        train_config: TrainConfig | None = None,
        split_config: SplitConfig | None = None,
        reporter: ReportWritter | None = None,
        random_seed: int = 42,
    ) -> None:
        """Bind the pipeline to its configuration objects."""
        self._dataset_config = dataset_config
        self._sequence_config = sequence_config or SequenceConfig()
        self._model_config = model_config or ModelConfig()
        self._train_config = train_config or TrainConfig()
        self._split_config = split_config or SplitConfig()
        self._schema = PaySimSchema()

        self._reporter = reporter or ReportWritter(
            dataset_config,
            domain_data="paysim_synthetic_financial",
            pipeline="model_training",
        )
        self._reporter.set_path_info_from_module()
        self._seed = random_seed

    def run(self) -> dict[str, object]:
        """Run the full pipeline once and return a JSON-friendly summary."""
        self._record_run_config([self._train_config.seed])
        prepared = self._prepare_data()
        summary = self._train_and_evaluate(prepared, self._train_config)
        self._report_single(summary)
        return summary

    def run_with_seeds(self, seeds: Sequence[int]) -> dict[str, object]:
        """Train/evaluate once per seed on the SAME data split, report mean/std.

        Data loading, scaling, and windowing are expensive and seed-independent,
        so they run once; only model init/shuffle/dropout vary per seed. This
        separates genuine improvement from run-to-run training noise.
        """
        self._record_run_config(list(seeds))
        prepared = self._prepare_data()
        per_seed: list[dict[str, object]] = []
        for seed in seeds:
            seed_train_config = replace(self._train_config, seed=seed)
            summary = self._train_and_evaluate(prepared, seed_train_config)
            summary["seed"] = seed
            per_seed.append(summary)
            self._reporter.display_text(
                f"[seed {seed}] best_epoch {summary['best_epoch']} | "
                f"val AUC {summary['best_val_auc']:.4f} | test AUC {summary['test_auc']:.4f} | "
                f"test AP {summary['test_ap']:.4f}"
            )

        aggregate = self._aggregate_seeds(per_seed)
        self._reporter.display_section(
            f"LSTM-Autoencoder multi-seed summary (W={self._sequence_config.window_length}, seeds={list(seeds)})"
        )
        self._reporter.display_text(
            f"test AUC {aggregate['test_auc_mean']:.4f} +/- {aggregate['test_auc_std']:.4f} | "
            f"test AP {aggregate['test_ap_mean']:.4f} +/- {aggregate['test_ap_std']:.4f}"
        )
        self._reporter.update_summary({"multi_seed_training": {"runs": per_seed, "aggregate": aggregate}})
        self._reporter.save_reports(
            text_filename="multi_seed_training_report.txt",
            json_filename="multi_seed_training_report.json",
        )
        return aggregate

    def _record_run_config(self, seeds: Sequence[int]) -> None:
        """Write the resolved run configuration into the report (JSON + transcript).

        Unlike the free-text `command` field (what was typed), this records
        what the pipeline is configured to train: architecture, window, seeds
        and the checkpoint file each seed will write.
        """
        config = self._model_config
        run_config = {
            "architecture": config.architecture_tag,
            "window_length": self._sequence_config.window_length,
            "seeds": list(seeds),
            "model_config": {
                "attention": config.attention,
                "bottleneck": config.bottleneck,
                "attention_pooling": config.attention_pooling,
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
            },
            "checkpoints": [self._checkpoint_name(seed) for seed in seeds],
        }
        self._reporter.update_summary({"run_config": run_config})
        self._reporter.display_text(
            f"Run config: architecture={config.architecture_tag} | "
            f"W={self._sequence_config.window_length} | seeds={list(seeds)}"
        )

    def prepared_test_data(self) -> tuple[TransactionSequenceDataset, np.ndarray, np.ndarray]:
        """Return (test windows, test labels, test row positions into `built`).

        For reuse by analyses that score an already-trained checkpoint -- e.g.
        correlating LSTM-AE reconstruction error with another model's score on
        the SAME test rows -- without repeating the full prepare+train run.
        """
        prepared = self._prepare_data()
        return prepared.test_subset, prepared.test_labels, prepared.test_row_index

    def prepared_scoring_splits(self) -> dict[str, SplitScoringData]:
        """Return train (ALL rows, normal and fraud), val and test windows with labels and row positions.

        Used to score a trained checkpoint on every split, e.g. to build a
        reconstruction-error feature for a hybrid classifier. Note the AE only
        trained on the normal rows of `train`, so its error there is in-sample.
        """
        prepared = self._prepare_data(with_train_scoring=True)
        return {
            "train": SplitScoringData(prepared.train_all_subset, prepared.train_all_labels, prepared.train_row_index),
            "val": SplitScoringData(prepared.val_subset, prepared.val_labels, prepared.val_row_index),
            "test": SplitScoringData(prepared.test_subset, prepared.test_labels, prepared.test_row_index),
        }

    def checkpoint_path(self, seed: int) -> Path:
        """Path of the checkpoint `_train_and_evaluate` writes for `seed` with this configuration."""
        checkpoint_dir = self._dataset_config.artifacts_dir / "model" / "paysim_synthetic_financial"
        return checkpoint_dir / self._checkpoint_name(seed)

    def _prepare_data(self, with_train_scoring: bool = False) -> _PreparedData:
        """Load, scale, and window the dataset once; reused across seeds.

        `with_train_scoring` additionally wraps ALL train rows (not just normal
        ones) in a dataset and keeps train/val row positions; off by default so
        training runs do not pay for the extra shared-memory copy.
        """
        loader = DatasetLoader(self._dataset_config, self._schema)
        loader._parse_timestamps = lambda series: series
        loader._normalize_target = lambda series: series
        loaded = loader.load()

        built = CausalFeatureBuilder(self._dataset_config, self._schema).build(loaded.frame)
        receiver_ids = self._split_receiver_ids(built)

        train_normal_mask = built[self._schema.receiver_column].isin(receiver_ids["train"]) & (
            built[self._schema.target_column] == 0
        )
        log1p_columns = (self._schema.amount_column, HISTORY_LENGTH)
        scaler = FeatureScaler(self._sequence_config.feature_columns, log1p_columns).fit(
            built, train_normal_mask
        )
        scaled = scaler.transform(built)

        batch = SequenceBuilder(self._sequence_config, self._schema).build(scaled)
        splits = self._rows_by_receiver(batch, receiver_ids)
        normal_train_idx = splits["train"][batch.labels[splits["train"]] == 0]

        return _PreparedData(
            train_subset=TransactionSequenceDataset(batch, normal_train_idx),
            val_subset=TransactionSequenceDataset(batch, splits["val"]),
            test_subset=TransactionSequenceDataset(batch, splits["test"]),
            val_labels=batch.labels[splits["val"]],
            test_labels=batch.labels[splits["test"]],
            test_row_index=splits["test"],
            n_train_normal=len(normal_train_idx),
            n_val=len(splits["val"]),
            n_test=len(splits["test"]),
            train_all_subset=TransactionSequenceDataset(batch, splits["train"]) if with_train_scoring else None,
            train_all_labels=batch.labels[splits["train"]] if with_train_scoring else None,
            train_row_index=splits["train"] if with_train_scoring else None,
            val_row_index=splits["val"] if with_train_scoring else None,
        )

    def _train_and_evaluate(
        self, prepared: _PreparedData, train_config: TrainConfig
    ) -> dict[str, object]:
        """Fit one model on `prepared` with `train_config` and score it on test."""
        checkpoint_dir = self._dataset_config.artifacts_dir / "model" / "paysim_synthetic_financial"
        train_config = replace(train_config, checkpoint_name=self._checkpoint_name(train_config.seed))
        trainer = Trainer(self._model_config, train_config, checkpoint_dir)
        model, result = trainer.fit(prepared.train_subset, prepared.val_subset, prepared.val_labels)
        test_auc, test_ap = trainer.score(model, prepared.test_subset, prepared.test_labels)

        return {
            "architecture": model.architecture_tag,
            "checkpoint_path": result.checkpoint_path,
            "window_length": self._sequence_config.window_length,
            "n_train_normal": prepared.n_train_normal,
            "n_val": prepared.n_val,
            "n_test": prepared.n_test,
            "best_epoch": result.best_epoch,
            "best_val_auc": result.best_val_auc,
            "test_auc": test_auc,
            "test_ap": test_ap,
            "history": result.history,
        }

    def _report_single(self, summary: dict[str, object]) -> None:
        """Write the single-run console/JSON report (same format as before)."""
        self._reporter.display_section(f"LSTM-Autoencoder training (W={self._sequence_config.window_length})")
        self._reporter.display_text(
            f"Best epoch {summary['best_epoch']} | val AUC {summary['best_val_auc']:.4f} | "
            f"test AUC {summary['test_auc']:.4f} | test AP {summary['test_ap']:.4f}"
        )
        self._reporter.update_summary({"training": summary})
        self._reporter.save_reports(
            text_filename="training_report.txt",
            json_filename="training_report.json",
        )

    def _checkpoint_name(self, seed: int) -> str:
        """Checkpoint filename tagged by window, architecture variant, and seed
        so different configurations never overwrite each other's checkpoint."""
        return f"lstm_ae_w{self._sequence_config.window_length}_{self._architecture_tag()}_seed{seed}_best.pt"

    def _architecture_tag(self) -> str:
        """Return a short label for the current model architecture."""
        config = self._model_config
        return config.architecture_tag

    @staticmethod
    def _aggregate_seeds(per_seed: list[dict[str, object]]) -> dict[str, object]:
        """Mean and std of val/test metrics across seeds (ddof=1 when n > 1)."""
        val_aucs = np.array([run["best_val_auc"] for run in per_seed])
        test_aucs = np.array([run["test_auc"] for run in per_seed])
        test_aps = np.array([run["test_ap"] for run in per_seed])
        ddof = 1 if len(per_seed) > 1 else 0
        return {
            "n_seeds": len(per_seed),
            "val_auc_mean": float(val_aucs.mean()),
            "val_auc_std": float(val_aucs.std(ddof=ddof)),
            "test_auc_mean": float(test_aucs.mean()),
            "test_auc_std": float(test_aucs.std(ddof=ddof)),
            "test_ap_mean": float(test_aps.mean()),
            "test_ap_std": float(test_aps.std(ddof=ddof)),
        }

    def _split_receiver_ids(self, built: pd.DataFrame) -> dict[str, pd.Index]:
        """Split receivers by split_config fractions, stratified by per-receiver fraud rate."""
        receiver_column = self._schema.receiver_column
        target_column = self._schema.target_column
        stats = built.groupby(receiver_column).agg(
            n=(target_column, "size"), n_fraud=(target_column, "sum")
        )
        stats["fraud_rate"] = stats["n_fraud"] / stats["n"]
        bins = pd.cut(
            stats["fraud_rate"], bins=FRAUD_RATE_BINS, labels=FRAUD_RATE_LABELS, include_lowest=True
        ).fillna(FRAUD_RATE_LABELS[0])

        train_ids, rest_ids = train_test_split(
            stats.index,
            test_size=1 - self._split_config.train_fraction,
            random_state=self._seed,
            stratify=bins,
        )
        val_ratio = self._split_config.validation_fraction / (1 - self._split_config.train_fraction)
        val_ids, test_ids = train_test_split(
            rest_ids, test_size=1 - val_ratio, random_state=self._seed, stratify=bins.loc[rest_ids]
        )
        return {"train": train_ids, "val": val_ids, "test": test_ids}

    def _rows_by_receiver(self, batch: SequenceBatch, receiver_ids: dict[str, pd.Index]) -> dict[str, np.ndarray]:
        """Map receiver-id splits onto sequence-batch row indices."""
        receiver_series = pd.Series(batch.receiver)
        return {
            name: np.flatnonzero(receiver_series.isin(ids).to_numpy())
            for name, ids in receiver_ids.items()
        }