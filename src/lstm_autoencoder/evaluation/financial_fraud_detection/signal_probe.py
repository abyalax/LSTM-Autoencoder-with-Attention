"""Check whether the fraud label can be learned from the available features."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.evaluation.config import ProbeConfig, SplitConfig
from lstm_autoencoder.data.financial_fraud_detection.causal_features import (
    ACCOUNT_CODE,
    CausalFeatureBuilder,
)
from lstm_autoencoder.data.financial_fraud_detection.schema import FinancialFraudDetectionSchema
from lstm_autoencoder.evaluation.signal_probe import BaseSignalProbe
from lstm_autoencoder.reporting.reporter import ReportWritter

if TYPE_CHECKING:
    from lstm_autoencoder.data.loader import DatasetLoader, LoadedDataset

CLUSTERING_THRESHOLD = 1.1
MIN_ACCOUNT_SIZE = 2


class SignalProbe(BaseSignalProbe):
    """Train a strong supervised model and compare it with a shuffled-label control.

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
        super().__init__(dataset_config, probe_config, split_config, loader, reporter)
        self._schema = FinancialFraudDetectionSchema()
        if self._loader is None:
            from lstm_autoencoder.data.loader import DatasetLoader

            self._loader = DatasetLoader(dataset_config, self._schema)
        self._builder = CausalFeatureBuilder(dataset_config, self._schema)

    def _get_schema(self) -> FinancialFraudDetectionSchema:
        """Return the schema for the dataset."""
        return self._schema

    def _prepare_data(self, loaded: LoadedDataset) -> tuple[pd.DataFrame, np.ndarray]:
        """Prepare the data for signal probe."""
        built = self._builder.build(loaded.frame)
        ordered = built.sort_values(self._schema.timestamp_column, kind="stable")
        ordered = ordered.reset_index(drop=True)
        labels = ordered[self._schema.target_column].to_numpy()
        return ordered, labels

    def _display_additional_info(self, ordered: pd.DataFrame, labels: np.ndarray) -> None:
        """Display account dispersion information."""
        self._display_account_dispersion(ordered, float(labels.mean()))

    def _display_account_dispersion(self, ordered: pd.DataFrame, rate: float) -> None:
        per_account = ordered.groupby(ACCOUNT_CODE)[self._schema.target_column]
        grouped = per_account.agg(["size", "sum"])
        grouped = grouped[grouped["size"] >= MIN_ACCOUNT_SIZE]
        expected = grouped["size"] * rate
        variance = grouped["size"] * rate * (1 - rate)
        chi_square = (((grouped["sum"] - expected) ** 2) / variance).sum()
        index = float(chi_square / len(grouped))
        self._reporter.display_text(
            f"\nIndeks dispersi fraud per akun = {index:.4f} (1.0 berarti fraud tersebar acak antar akun)"
        )
        if index > CLUSTERING_THRESHOLD:
            self._reporter.display_text("  -> ada akun yang lebih berisiko.")

    def _build_features(self, ordered: pd.DataFrame) -> tuple[pd.DataFrame, list[bool]]:
        excluded = {
            self._schema.target_column,
            self._schema.id_column,
            self._schema.timestamp_column,
            self._schema.sender_column,
            self._schema.receiver_column,
            ACCOUNT_CODE,
            *self._schema.leakage_columns,
            *self._schema.reuse_columns,
        }
        derived = self._builder.feature_columns(ordered)
        features = pd.DataFrame(index=ordered.index)
        categorical: list[bool] = []
        for column in ordered.columns:
            if column in excluded or column in derived:
                continue
            series = ordered[column]
            if pd.api.types.is_numeric_dtype(series):
                features[column] = series
                categorical.append(False)
            elif series.nunique() <= self._probe.max_categories:
                features[column] = pd.factorize(series)[0].astype("int32")
                categorical.append(True)
        for column in derived:
            features[column] = ordered[column]
            categorical.append(False)
        for column in self._schema.reuse_columns:
            if column in ordered.columns:
                for name, values in self._reuse_counts(ordered, column).items():
                    features[name] = values
                    categorical.append(False)
        return features, categorical

    @staticmethod
    def _reuse_counts(ordered: pd.DataFrame, column: str) -> dict[str, pd.Series]:
        codes = pd.Series(pd.factorize(ordered[column])[0], index=ordered.index)
        usage = codes.map(codes.value_counts())
        accounts = pd.DataFrame({"code": codes, "account": ordered[ACCOUNT_CODE]})
        distinct = accounts.groupby("code")["account"].nunique()
        return {
            f"{column}_tx_count": usage,
            f"{column}_n_accounts": codes.map(distinct),
        }
