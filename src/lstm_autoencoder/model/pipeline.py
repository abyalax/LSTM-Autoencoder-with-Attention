"""End-to-end training pipeline: load -> causal features -> scale -> sequences -> train -> evaluate."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.loader import DatasetLoader
from lstm_autoencoder.data.paysim_synthetic_financial.causal_features import CausalFeatureBuilder
from lstm_autoencoder.data.paysim_synthetic_financial.schema import PaySimSchema
from lstm_autoencoder.model.config import ModelConfig, SequenceConfig, TrainConfig
from lstm_autoencoder.model.dataset import TransactionSequenceDataset
from lstm_autoencoder.model.feature_scaler import FeatureScaler
from lstm_autoencoder.model.sequence_builder import SequenceBatch, SequenceBuilder
from lstm_autoencoder.model.trainer import Trainer
from lstm_autoencoder.reporting.reporter import ReportWritter

FRAUD_RATE_BINS: tuple[float, ...] = (0, 0.001, 0.01, 0.1, 1.0)
FRAUD_RATE_LABELS: tuple[str, ...] = ("very_low", "low", "medium", "high")


class PaySimModelTrainingPipeline:
    """Orchestrate loading, scaling, sequence building, training, and evaluation."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        sequence_config: SequenceConfig | None = None,
        model_config: ModelConfig | None = None,
        train_config: TrainConfig | None = None,
        reporter: ReportWritter | None = None,
        random_seed: int = 42,
    ) -> None:
        """Bind the pipeline to its configuration objects."""
        self._dataset_config = dataset_config
        self._sequence_config = sequence_config or SequenceConfig()
        self._model_config = model_config or ModelConfig()
        self._train_config = train_config or TrainConfig()
        self._schema = PaySimSchema()
        self._reporter = reporter or ReportWritter(
            dataset_config, domain_data="paysim_synthetic_financial", pipeline="model_training"
        )
        self._reporter.set_path_info_from_module()
        self._seed = random_seed

    def run(self) -> dict[str, object]:
        """Run the full pipeline and return a JSON-friendly summary."""
        loader = DatasetLoader(self._dataset_config, self._schema)
        loader._parse_timestamps = lambda series: series
        loader._normalize_target = lambda series: series
        loaded = loader.load()

        built = CausalFeatureBuilder(self._dataset_config, self._schema).build(loaded.frame)
        receiver_ids = self._split_receiver_ids(built)

        train_normal_mask = built[self._schema.receiver_column].isin(receiver_ids["train"]) & (
            built[self._schema.target_column] == 0
        )
        scaler = FeatureScaler(self._sequence_config.feature_columns).fit(built, train_normal_mask)
        scaled = scaler.transform(built)

        batch = SequenceBuilder(self._sequence_config, self._schema).build(scaled)
        splits = self._rows_by_receiver(batch, receiver_ids)

        normal_train_idx = splits["train"][batch.labels[splits["train"]] == 0]
        train_subset = TransactionSequenceDataset(batch, normal_train_idx)
        val_subset = TransactionSequenceDataset(batch, splits["val"])
        test_subset = TransactionSequenceDataset(batch, splits["test"])

        checkpoint_dir = self._dataset_config.artifacts_dir / "model" / "paysim_synthetic_financial"
        window_train_config = replace(
            self._train_config,
            checkpoint_name=f"lstm_ae_w{self._sequence_config.window_length}_best.pt",
        )
        trainer = Trainer(self._model_config, window_train_config, checkpoint_dir)
        model, result = trainer.fit(train_subset, val_subset, batch.labels[splits["val"]])
        test_auc, test_ap = trainer.score(model, test_subset, batch.labels[splits["test"]])

        summary = {
            "window_length": self._sequence_config.window_length,
            "n_train_normal": len(normal_train_idx),
            "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
            "best_epoch": result.best_epoch,
            "best_val_auc": result.best_val_auc,
            "test_auc": test_auc,
            "test_ap": test_ap,
            "history": result.history,
        }
        self._reporter.display_section(f"LSTM-Autoencoder training (W={self._sequence_config.window_length})")
        self._reporter.display_text(
            f"Best epoch {result.best_epoch} | val AUC {result.best_val_auc:.4f} | "
            f"test AUC {test_auc:.4f} | test AP {test_ap:.4f}"
        )
        self._reporter.update_summary({"training": summary})
        self._reporter.save_reports(
            text_filename="training_report.txt",
            json_filename="training_report.json",
        )
        return summary

    def _split_receiver_ids(self, built: pd.DataFrame) -> dict[str, pd.Index]:
        """Split receivers 70/15/15, stratified by per-receiver fraud rate."""
        receiver_column = self._schema.receiver_column
        target_column = self._schema.target_column
        stats = built.groupby(receiver_column).agg(
            n=(target_column, "size"), n_fraud=(target_column, "sum")
        )
        stats["fraud_rate"] = stats["n_fraud"] / stats["n"]
        bins = pd.cut(
            stats["fraud_rate"], bins=FRAUD_RATE_BINS, labels=FRAUD_RATE_LABELS, include_lowest=True
        ).fillna(FRAUD_RATE_LABELS[0])

        train_ids, rest_ids = train_test_split(stats.index, test_size=0.30, random_state=self._seed, stratify=bins)
        val_ids, test_ids = train_test_split(
            rest_ids, test_size=0.50, random_state=self._seed, stratify=bins.loc[rest_ids]
        )
        return {"train": train_ids, "val": val_ids, "test": test_ids}

    def _rows_by_receiver(self, batch: SequenceBatch, receiver_ids: dict[str, pd.Index]) -> dict[str, np.ndarray]:
        """Map receiver-id splits onto sequence-batch row indices."""
        receiver_series = pd.Series(batch.receiver)
        return {
            name: np.flatnonzero(receiver_series.isin(ids).to_numpy())
            for name, ids in receiver_ids.items()
        }