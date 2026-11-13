"""End-to-end training pipeline for BankSim: load -> causal features -> scale -> sequences -> train -> evaluate."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.banksim.ae_features import (
    CONTINUOUS_FEATURES,
    LOG1P_FEATURES,
    add_ae_features,
)
from lstm_autoencoder.data.banksim.causal_features import (
    CausalFeatureBuilder,
    account_identity,
)
from lstm_autoencoder.data.banksim.schema import BankSimSchema
from lstm_autoencoder.data.loader import DatasetLoader
from lstm_autoencoder.evaluation.config import SplitConfig
from lstm_autoencoder.evaluation.banksim.split import BANKSIM_SPLIT_CONFIG, split_customers
from lstm_autoencoder.model.config import ModelConfig, SequenceConfig, TrainConfig
from lstm_autoencoder.model.dataset import TransactionSequenceDataset
from lstm_autoencoder.model.feature_scaler import FeatureScaler
from lstm_autoencoder.model.lstm_autoencoder import LSTMAutoencoder
from lstm_autoencoder.model.revision import PIPELINE_REVISION, REVISION_KEY
from lstm_autoencoder.model.sequence_builder import SequenceBatch, SequenceBuilder
from lstm_autoencoder.model.trainer import Trainer
from lstm_autoencoder.reporting.reporter import ReportWritter


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
    train_history_bucket: np.ndarray | None = None
    val_history_bucket: np.ndarray | None = None
    test_history_bucket: np.ndarray | None = None
    val_receiver: np.ndarray | None = None
    test_receiver: np.ndarray | None = None


@dataclass(frozen=True)
class SplitScoringData:
    """Windows, labels and batch-row positions of one split, for scoring a trained checkpoint."""

    dataset: TransactionSequenceDataset
    labels: np.ndarray
    row_index: np.ndarray
    history_bucket: np.ndarray | None = None
    receiver: np.ndarray | None = None  # receiver id per row, for receiver-cluster resampling


class BankSimModelTrainingPipeline:
    """Orchestrate loading, scaling, sequence building, training, and evaluation for BankSim."""

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
        self._split_config = split_config or BANKSIM_SPLIT_CONFIG
        self._schema = BankSimSchema()
        # Windows are built per CUSTOMER. `SequenceBuilder` keys its windows (and the
        # `receiver` of every example) on `receiver_column`, which is `merchant` in the
        # BankSim schema, so point it at the customer column instead.
        self._sequence_schema = replace(self._schema, receiver_column=self._schema.sender_column)

        self._reporter = reporter or ReportWritter(
            dataset_config,
            domain_data="banksim",
            pipeline="model_training",
        )
        self._reporter.set_path_info_from_module()
        self._seed = random_seed
        # Allocate one immutable artifact directory per training invocation.
        # This deliberately does not use timestamps: versions are easy to find,
        # compare, and never overwrite an earlier experiment.
        self._artifact_version_dir: Path | None = None

    @property
    def sequence_config(self) -> SequenceConfig:
        """Sequence config with the BankSim feature columns (final after `_prepare_data` ran)."""
        return self._sequence_config

    @property
    def model_config(self) -> ModelConfig:
        """Model config with `input_dim` matching the feature columns (final after `_prepare_data` ran)."""
        return self._model_config

    def _resolve_features(self, columns: tuple[str, ...]) -> None:
        """Adopt the BankSim feature columns and size the model input to match.

        The number of one-hot category columns comes from the data, so the input
        dimension is only known once the data is loaded. Whatever `feature_columns` /
        `input_dim` the caller passed (the PaySim defaults) is replaced here.
        """
        self._sequence_config = replace(self._sequence_config, feature_columns=columns)
        self._model_config = replace(self._model_config, input_dim=len(columns))

    def run(self) -> dict[str, object]:
        """Run the full pipeline once and return a JSON-friendly summary."""
        prepared = self._prepare_data()  # first: it fixes the feature columns / input_dim
        self._record_run_config([self._train_config.seed])
        summary = self._train_and_evaluate(prepared, self._train_config)
        self._report_single(summary)
        return summary

    def run_with_seeds(self, seeds: Sequence[int]) -> dict[str, object]:
        """Train/evaluate once per seed on the SAME data split, report mean/std.

        Data loading, scaling, and windowing are expensive and seed-independent,
        so they run once; only model init/shuffle/dropout vary per seed. This
        separates genuine improvement from run-to-run training noise.
        """
        prepared = self._prepare_data()  # first: it fixes the feature columns / input_dim
        self._record_run_config(list(seeds))
        per_seed: list[dict[str, object]] = []
        for seed in seeds:
            seed_train_config = replace(self._train_config, seed=seed)
            summary = self._train_and_evaluate(prepared, seed_train_config)
            summary["seed"] = seed
            per_seed.append(summary)
            self._reporter.display_text(
                f"[seed {seed}] best_epoch {summary['best_epoch']} ({summary['selection_metric']}) | "
                f"val loss {summary['best_val_loss']:.5f} | val AUC {summary['best_val_auc']:.4f} "
                f"(peak {summary['peak_val_auc']:.4f} @ep{summary['peak_val_auc_epoch']}) | "
                f"test AUC {summary['test_auc']:.4f} | test AP {summary['test_ap']:.4f} | "
                + self._recall_text(summary, prefix="test_recall_at_fpr_")
            )

        aggregate = self._aggregate_seeds(per_seed)
        self._reporter.display_section(
            f"LSTM-Autoencoder multi-seed summary (W={self._sequence_config.window_length}, seeds={list(seeds)})"
        )
        self._reporter.display_text(
            f"val loss {aggregate['val_loss_mean']:.5f} +/- {aggregate['val_loss_std']:.5f} | "
            f"val AUC {aggregate['val_auc_mean']:.4f} +/- {aggregate['val_auc_std']:.4f} "
            f"(peak {aggregate['peak_val_auc_mean']:.4f})"
        )
        self._reporter.display_text(
            f"test AUC {aggregate['test_auc_mean']:.4f} +/- {aggregate['test_auc_std']:.4f} | "
            f"test AP {aggregate['test_ap_mean']:.4f} +/- {aggregate['test_ap_std']:.4f} | "
            + self._recall_text(aggregate, prefix="test_recall_at_fpr_", mean_std=True)
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
        if self._artifact_version_dir is None:
            self._artifact_version_dir = self._new_artifact_version_dir()
        config = self._model_config
        run_config = {
            "artifact_version": self._artifact_version_dir.name,
            REVISION_KEY: PIPELINE_REVISION,
            "selection_metric": self._train_config.selection_metric,
            "architecture": config.architecture_tag,
            "window_length": self._sequence_config.window_length,
            "seeds": list(seeds),
            "feature_columns": list(self._sequence_config.feature_columns),
            "model_config": {
                "input_dim": config.input_dim,
                "attention": config.attention,
                "bottleneck": config.bottleneck,
                "attention_pooling": config.attention_pooling,
                "hidden_dim": config.hidden_dim,
                "bottleneck_dim": config.bottleneck_dim,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
                "decoder": "lstm",
                "parameter_count": self._parameter_count(),
            },
            "train_config": {
                "optimizer": "adam",
                "batch_size": self._train_config.batch_size,
                "max_epochs": self._train_config.max_epochs,
                "learning_rate": self._train_config.learning_rate,
                "weight_decay": self._train_config.weight_decay,
                "patience": self._train_config.patience,
                "min_delta": self._train_config.min_delta,
                "selection_metric": self._train_config.selection_metric,
                "device": self._train_config.device,
            },
            "checkpoints": [self._checkpoint_name(seed) for seed in seeds],
        }
        self._reporter.update_summary({"run_config": run_config})
        self._reporter.display_text(
            f"Run config: architecture={config.architecture_tag} | "
            f"W={self._sequence_config.window_length} | seeds={list(seeds)}"
        )
        metadata_path = self._artifact_version_dir / "metadata.json"
        metadata_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        (self._artifact_version_dir.parent / "latest.json").write_text(
            json.dumps({"version": self._artifact_version_dir.name}, indent=2), encoding="utf-8"
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
            "train": SplitScoringData(
                prepared.train_all_subset,
                prepared.train_all_labels,
                prepared.train_row_index,
                prepared.train_history_bucket,
            ),
            "val": SplitScoringData(
                prepared.val_subset, prepared.val_labels, prepared.val_row_index, prepared.val_history_bucket
            ),
            "test": SplitScoringData(
                prepared.test_subset, prepared.test_labels, prepared.test_row_index, prepared.test_history_bucket
            ),
        }

    def prepared_eval_splits(self) -> dict[str, SplitScoringData]:
        """Return validation and test windows with labels, row positions and history buckets.

        Lighter than `prepared_scoring_splits` (no copy of the train rows), for
        diagnostics that only score a trained checkpoint on held-out data.
        """
        prepared = self._prepare_data()
        return {
            "val": SplitScoringData(
                prepared.val_subset,
                prepared.val_labels,
                prepared.val_row_index,
                prepared.val_history_bucket,
                prepared.val_receiver,
            ),
            "test": SplitScoringData(
                prepared.test_subset,
                prepared.test_labels,
                prepared.test_row_index,
                prepared.test_history_bucket,
                prepared.test_receiver,
            ),
        }

    def checkpoint_path(self, seed: int) -> Path:
        """Path of the checkpoint `_train_and_evaluate` writes for `seed` with this configuration.

        While training, that is inside this run's own version directory. For
        evaluation-only use, the NEWEST version directory that holds a checkpoint of
        this exact name wins, because each training run gets its own version
        directory and `latest.json` only points at the last one trained.
        """
        name = self._checkpoint_name(seed)
        if self._artifact_version_dir is not None:
            return self._artifact_version_dir / name
        found = self.find_checkpoint(self._dataset_config.artifacts_dir, name)
        return found if found is not None else self._latest_artifact_version_dir() / name

    @staticmethod
    def find_checkpoint(artifacts_dir: Path, name: str) -> Path | None:
        """Newest `version_N_*` directory containing checkpoint `name`, else the legacy flat folder, else None."""
        root = artifacts_dir / "model" / "banksim"
        if not root.exists():
            return None
        pattern = re.compile(r"^version_(\d+)(?:_.*)?$")
        versions = sorted(
            (
                (int(match.group(1)), child)
                for child in root.iterdir()
                if child.is_dir() and (match := pattern.match(child.name))
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        for _, directory in versions:
            if (directory / name).exists():
                return directory / name
        legacy = root / name
        return legacy if legacy.exists() else None

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

        train_normal_mask = account_identity(built, self._schema).isin(receiver_ids["train"]) & (
            built[self._schema.target_column] == 0
        )
        # Category one-hots and the per-category amount z-score; the z-score statistics are
        # learned from train-normal rows only. The built frame keeps its row order.
        built, feature_columns = add_ae_features(built, train_normal_mask, self._schema)
        self._resolve_features(feature_columns)
        indicator_columns = tuple(c for c in feature_columns if c not in CONTINUOUS_FEATURES)
        scaler = FeatureScaler(feature_columns, LOG1P_FEATURES, passthrough_columns=indicator_columns).fit(
            built, train_normal_mask
        )
        scaled = scaler.transform(built)

        batch = SequenceBuilder(self._sequence_config, self._sequence_schema).build(scaled)
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
            train_history_bucket=batch.history_bucket[splits["train"]] if with_train_scoring else None,
            val_history_bucket=batch.history_bucket[splits["val"]],
            test_history_bucket=batch.history_bucket[splits["test"]],
            val_receiver=batch.receiver[splits["val"]],
            test_receiver=batch.receiver[splits["test"]],
        )

    def _train_and_evaluate(self, prepared: _PreparedData, train_config: TrainConfig) -> dict[str, object]:
        """Fit one model on `prepared` with `train_config` and score it on test."""
        if self._artifact_version_dir is None:
            raise RuntimeError("Training artifact directory has not been initialized")
        checkpoint_dir = self._artifact_version_dir
        train_config = replace(train_config, checkpoint_name=self._checkpoint_name(train_config.seed))
        trainer = Trainer(self._model_config, train_config, checkpoint_dir)
        model, result = trainer.fit(prepared.train_subset, prepared.val_subset, prepared.val_labels)
        test_metrics = trainer.score(model, prepared.test_subset, prepared.test_labels)

        summary = {
            "architecture": model.architecture_tag,
            "checkpoint_path": result.checkpoint_path,
            "window_length": self._sequence_config.window_length,
            "n_train_normal": prepared.n_train_normal,
            "n_val": prepared.n_val,
            "n_test": prepared.n_test,
            "best_epoch": result.best_epoch,
            "selection_metric": result.selection_metric,
            "best_val_loss": result.best_val_loss,
            "best_val_auc": result.best_val_auc,
            "peak_val_auc": result.peak_val_auc,
            "peak_val_auc_epoch": result.peak_val_auc_epoch,
            "test_auc": test_metrics["auc"],
            "test_ap": test_metrics["ap"],
            "history": result.history,
        }
        # Fold recall@FPR_x into the summary as top-level `test_recall_at_fpr_x` fields
        for key, value in test_metrics.items():
            if key.startswith("recall_at_fpr_"):
                summary[f"test_{key}"] = value
        return summary

    def _report_single(self, summary: dict[str, object]) -> None:
        """Write the single-run console/JSON report (same format as before)."""
        self._reporter.display_section(f"LSTM-Autoencoder training (W={self._sequence_config.window_length})")
        self._reporter.display_text(
            f"Best epoch {summary['best_epoch']} ({summary['selection_metric']}) | "
            f"val loss {summary['best_val_loss']:.5f} | val AUC {summary['best_val_auc']:.4f} "
            f"(peak {summary['peak_val_auc']:.4f} @ep{summary['peak_val_auc_epoch']}) | "
            f"test AUC {summary['test_auc']:.4f} | test AP {summary['test_ap']:.4f} | "
            + self._recall_text(summary, prefix="test_recall_at_fpr_")
        )
        self._reporter.update_summary({"training": summary})
        self._reporter.save_reports(
            text_filename="training_report.txt",
            json_filename="training_report.json",
        )

    @staticmethod
    def _recall_text(source: dict[str, object], prefix: str, mean_std: bool = False) -> str:
        """Format every `{prefix}{fpr}[_mean/_std]` field as `recall@1%=0.1234` (or `0.1234 +/- 0.0056`).

        Keeps the FPR target itself out of the format string: whatever
        `TrainConfig.fpr_targets` a run used, its recall fields show up here
        without the console text needing to hardcode 1%/0.1%.
        """
        if mean_std:
            keys = sorted({k[: -len("_mean")] for k in source if k.startswith(prefix) and k.endswith("_mean")})
            parts = []
            for key in keys:
                fpr_pct = float(key[len(prefix) :]) * 100
                parts.append(f"recall@{fpr_pct:g}%={source[f'{key}_mean']:.4f}+/-{source[f'{key}_std']:.4f}")
            return " | ".join(parts)
        keys = sorted(k for k in source if k.startswith(prefix))
        parts = []
        for key in keys:
            fpr_pct = float(key[len(prefix) :]) * 100
            parts.append(f"recall@{fpr_pct:g}%={source[key]:.4f}")
        return " | ".join(parts)

    @staticmethod
    def _aggregate_seeds(per_seed: list[dict[str, object]]) -> dict[str, object]:
        """Mean and std of val/test metrics across seeds (ddof=1 when n > 1)."""
        val_losses = np.array([run["best_val_loss"] for run in per_seed])
        val_aucs = np.array([run["best_val_auc"] for run in per_seed])
        peak_aucs = np.array([run["peak_val_auc"] for run in per_seed])
        test_aucs = np.array([run["test_auc"] for run in per_seed])
        test_aps = np.array([run["test_ap"] for run in per_seed])
        ddof = 1 if len(per_seed) > 1 else 0
        aggregate: dict[str, object] = {
            "n_seeds": len(per_seed),
            "val_loss_mean": float(val_losses.mean()),
            "val_loss_std": float(val_losses.std(ddof=ddof)),
            "val_auc_mean": float(val_aucs.mean()),
            "val_auc_std": float(val_aucs.std(ddof=ddof)),
            "peak_val_auc_mean": float(peak_aucs.mean()),
            "test_auc_mean": float(test_aucs.mean()),
            "test_auc_std": float(test_aucs.std(ddof=ddof)),
            "test_ap_mean": float(test_aps.mean()),
            "test_ap_std": float(test_aps.std(ddof=ddof)),
        }
        # Aggregate every `test_recall_at_fpr_*` field the same way
        recall_keys = sorted({key for run in per_seed for key in run if key.startswith("test_recall_at_fpr_")})
        for key in recall_keys:
            values = np.array([run[key] for run in per_seed])
            aggregate[f"{key}_mean"] = float(values.mean())
            aggregate[f"{key}_std"] = float(values.std(ddof=ddof))
        return aggregate

    def _new_artifact_version_dir(self) -> Path:
        """Create the next numbered, metric-labelled model directory atomically."""
        root = self._dataset_config.artifacts_dir / "model" / "banksim"
        root.mkdir(parents=True, exist_ok=True)
        pattern = re.compile(r"^version_(\d+)(?:_.*)?$")
        numbers = [
            int(match.group(1)) for child in root.iterdir() if child.is_dir() and (match := pattern.match(child.name))
        ]
        next_number = max(numbers, default=0) + 1
        suffix = "auc" if self._train_config.selection_metric == "val_auc" else "loss"
        version_dir = root / f"version_{next_number}_{suffix}"
        version_dir.mkdir()
        return version_dir

    def _latest_artifact_version_dir(self) -> Path:
        root = self._dataset_config.artifacts_dir / "model" / "banksim"
        manifest = root / "latest.json"
        if manifest.exists():
            name = json.loads(manifest.read_text(encoding="utf-8"))["version"]
            directory = root / name
            if directory.is_dir():
                return directory
        # Compatibility with checkpoints produced before versioned artifacts.
        return root

    def _checkpoint_name(self, seed: int) -> str:
        """Checkpoint filename tagged by window, architecture variant, and seed
        so different configurations never overwrite each other's checkpoint."""
        return f"lstm_ae_w{self._sequence_config.window_length}_{self._architecture_tag()}_seed{seed}_best.pt"

    def _architecture_tag(self) -> str:
        """Return a short label for the current model architecture."""
        config = self._model_config
        return config.architecture_tag

    def _parameter_count(self) -> int:
        """Trainable parameter count of the configured architecture, CPU-only, for the run-config report.

        Builds a throwaway (never-moved-to-device, never-trained) instance
        purely to read `LSTMAutoencoder.parameter_count`; cheap relative to
        training and avoids duplicating the parameter-counting formula here.
        """
        config = self._model_config
        model = LSTMAutoencoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            attention=config.attention,
            bottleneck=config.bottleneck,
            attention_pooling=config.attention_pooling,
            dropout=config.dropout,
            bottleneck_dim=config.bottleneck_dim,
        )
        return model.parameter_count

    def split_receivers(self, built: pd.DataFrame) -> dict[str, pd.Index]:
        """Receiver ids of the train/val/test split this pipeline uses (public for audits)."""
        return self._split_receiver_ids(built)

    def _split_receiver_ids(self, built: pd.DataFrame) -> dict[str, pd.Index]:
        """Split CUSTOMERS by split_config fractions WITHOUT reading any label.

        The method keeps its PaySim-era name ("receiver") because the shared audit and
        hybrid code call it; for BankSim the split unit is the customer, the same unit the
        windows and the ablation baselines use.
        """
        return split_customers(account_identity(built, self._schema), self._split_config, self._seed)

    def _rows_by_receiver(self, batch: SequenceBatch, receiver_ids: dict[str, pd.Index]) -> dict[str, np.ndarray]:
        """Map receiver-id splits onto sequence-batch row indices."""
        receiver_series = pd.Series(batch.receiver)
        return {name: np.flatnonzero(receiver_series.isin(ids).to_numpy()) for name, ids in receiver_ids.items()}
