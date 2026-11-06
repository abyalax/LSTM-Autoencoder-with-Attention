"""Reusable signal probe base class for checking label learnability."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.evaluation.config import ProbeConfig, SplitConfig
from lstm_autoencoder.evaluation.metrics import RankingMetrics
from lstm_autoencoder.reporting.reporter import ReportWritter

if TYPE_CHECKING:
    from lstm_autoencoder.data.loader import DatasetLoader, LoadedDataset


class SchemaProtocol(Protocol):
    """Protocol for dataset schemas."""

    target_column: str
    timestamp_column: str
    id_column: str
    sender_column: str
    receiver_column: str
    leakage_columns: tuple[str, ...]
    reuse_columns: tuple[str, ...]





class SignalVerdict(Enum):
    """Outcome of the learnability check."""

    NO_SIGNAL = "no_signal"
    WEAK_SIGNAL = "weak_signal"
    LEARNABLE = "learnable"


class BaseSignalProbe(ABC):
    """Base class for training a strong supervised model to check label learnability.

    If the best supervised model cannot beat chance, no model can detect fraud
    in this dataset, including the proposed LSTM autoencoder.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        probe_config: ProbeConfig | None = None,
        split_config: SplitConfig | None = None,
        loader: DatasetLoader | None = None,
        reporter: ReportWritter | None = None,
    ) -> None:
        """Create a probe; optional collaborators default to standard ones."""
        self._dataset_config = dataset_config
        self._probe = probe_config or ProbeConfig()
        self._split_config = split_config or SplitConfig()
        self._loader = loader
        self._reporter = reporter or ReportWritter(dataset_config)
        self._reporter.set_path_info_from_module()

    @abstractmethod
    def _get_schema(self) -> SchemaProtocol:
        """Return the schema for the dataset."""
        pass

    @abstractmethod
    def _prepare_data(self, loaded: LoadedDataset) -> tuple[pd.DataFrame, np.ndarray]:
        """Prepare the data for signal probe.

        Args:
            loaded: Loaded dataset from loader

        Returns:
            Tuple of (features DataFrame, labels array)
        """
        pass

    @abstractmethod
    def _build_features(
        self, ordered: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[bool]]:
        """Build features from the ordered dataframe.

        Args:
            ordered: DataFrame sorted chronologically

        Returns:
            Tuple of (features DataFrame, categorical mask list)
        """
        pass

    @abstractmethod
    def _display_additional_info(self, ordered: pd.DataFrame, labels: np.ndarray) -> None:
        """Display domain-specific information during probe run.

        Args:
            ordered: DataFrame sorted chronologically
            labels: Target labels
        """
        pass

    def run(self) -> SignalVerdict:
        """Run the probe, save its report, and return the verdict."""
        if self._loader is None:
            from lstm_autoencoder.data.loader import DatasetLoader
            self._loader = DatasetLoader(self._config, self._get_schema())

        loaded = self._loader.load()
        features, labels = self._prepare_data(loaded)

        self._reporter.display_section("PROBE SINYAL: apakah target bisa dipelajari?")
        self._reporter.display_text(
            f"Baris {len(features):,} | target {int(labels.sum()):,} "
            f"({labels.mean() * 100:.3f}%)"
        )

        self._display_additional_info(features, labels)
        feature_df, categorical = self._build_features(features)
        names = list(feature_df.columns)
        self._reporter.display_text(f"\nFitur ({len(names)}): {names}")
        results = self._evaluate(feature_df, categorical, labels)
        verdict = self._decide(results)
        self._reporter.save_reports(
            text_filename="signal_report.txt",
            json_filename="signal_report.json",
        )
        return verdict

    def _evaluate(
        self, features: pd.DataFrame, categorical: list[bool], labels: np.ndarray
    ) -> dict[str, dict[str, float]]:
        """Evaluate the features using supervised model and shuffled control."""
        n_rows = len(features)
        first = int(self._split_config.train_fraction * n_rows)
        second = int((self._split_config.train_fraction + self._split_config.validation_fraction) * n_rows)
        rng = np.random.default_rng(self._probe.random_seed)
        train = np.arange(first)
        if len(train) > self._probe.max_train_rows:
            chosen = rng.choice(train, self._probe.max_train_rows, replace=False)
            train = np.sort(chosen)
        splits = {"val": np.arange(first, second), "test": np.arange(second, n_rows)}

        self._reporter.display_text(
            f"\nBatas atas supervised | train {len(train):,} "
            f"val {len(splits['val']):,} test {len(splits['test']):,}"
        )
        shuffled = rng.permutation(labels[train])
        results = {
            "label asli": self._fit_and_score(
                features, categorical, train, labels[train], splits, labels
            ),
            "label diacak": self._fit_and_score(
                features, categorical, train, shuffled, splits, labels
            ),
        }
        rows = {
            name: {key: round(value, 4) for key, value in scores.items()}
            for name, scores in results.items()
        }
        self._reporter.display_table(pd.DataFrame(rows).T)
        return results

    def _fit_and_score(
        self,
        features: pd.DataFrame,
        categorical: list[bool],
        train: np.ndarray,
        train_labels: np.ndarray,
        splits: dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> dict[str, float]:
        """Fit model and compute metrics on validation and test sets."""
        model = HistGradientBoostingClassifier(
            max_iter=self._probe.max_iterations,
            learning_rate=self._probe.learning_rate,
            max_leaf_nodes=self._probe.max_leaf_nodes,
            early_stopping=False,
            categorical_features=categorical if any(categorical) else None,
            random_state=self._probe.random_seed,
        )
        model.fit(features.iloc[train], train_labels)
        scores: dict[str, float] = {}
        for name, index in splits.items():
            predicted = model.predict_proba(features.iloc[index])[:, 1]
            truth = labels[index]
            scores[f"{name}_auc"] = RankingMetrics.roc_auc(predicted, truth)
            scores[f"{name}_ap"] = RankingMetrics.average_precision(predicted, truth)
            scores[f"{name}_prevalence"] = float(truth.mean())
        return scores

    def _decide(self, results: dict[str, dict[str, float]]) -> SignalVerdict:
        """Decide the verdict based on evaluation results."""
        real = results["label asli"]
        control_auc = results["label diacak"]["test_auc"]
        auc, ap = real["test_auc"], real["test_ap"]
        prevalence = real["test_prevalence"]
        reporter = self._reporter.reporter
        reporter.display_text("\nKESIMPULAN")
        summary = (
            f"AUC test {auc:.3f}, AP {ap:.4f} (prevalensi {prevalence:.4f}), "
            f"kontrol label-acak AUC {control_auc:.3f}."
        )
        lift_limit = self._probe.no_signal_lift * prevalence
        if auc < self._probe.no_signal_auc and ap < lift_limit:
            reporter.display_text(f"  {summary}")
            reporter.display_text(
                "  Tidak ada sinyal yang dapat dipelajari. Dataset ini tidak layak "
                "untuk mengklaim kemampuan deteksi apa pun."
            )
            return SignalVerdict.NO_SIGNAL
        if auc < self._probe.weak_signal_auc:
            reporter.display_text(f"  {summary}")
            reporter.display_text(
                "  Sinyal sangat lemah, hasil model deep kemungkinan tidak bermakna."
            )
            return SignalVerdict.WEAK_SIGNAL
        reporter.display_text(f"  {summary}")
        reporter.display_text(
            "  Ada sinyal yang dapat dipelajari, lanjut ke desain model."
        )
        return SignalVerdict.LEARNABLE
