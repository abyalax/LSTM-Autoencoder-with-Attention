"""Base dataset exploratory with common exploratory logic for all fraud detection datasets.

This module provides a comprehensive exploratory that answers:
1. Entity cardinality - How many unique customer/accounts?
2. History distribution - How many transactions per account?
3. Fraud distribution - How many frauds occur on accounts with history?
4. Fraud position - Does fraud appear more often as first or subsequent transaction?
5. History before fraud - How many normal transactions before fraud?
6. Sequence length - Are there enough sequences with length 3, 5, 10, etc?
7. Temporal distribution - Is fraud distributed throughout the dataset period?

Domain-specific exploratories should inherit from this class and override
methods as needed for their specific dataset characteristics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.loader import DatasetLoader
from lstm_autoencoder.reporting.reporter import ReportWritter


@dataclass(frozen=True)
class ExploratoryConfig:
    """Configuration for dataset exploratory."""

    window_lengths: tuple[int, ...] = (1, 3, 5, 10, 15, 20)
    history_strata_bins: tuple[int, ...] = (0, 1, 4, 9, 19, np.inf)
    history_strata_labels: tuple[str, ...] = ("1", "2-4", "5-9", "10-19", ">=20")


class DatasetExploratory(ABC):
    """Base exploratory for all fraud detection datasets.

    Subclasses should:
    1. Implement the abstract `get_schema()` method to return their domain-specific schema
    2. Optionally override exploratory methods for domain-specific logic
    3. Optionally override reporting methods for domain-specific formatting
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        exploratory_config: ExploratoryConfig | None = None,
        loader: DatasetLoader | None = None,
        reporter: ReportWritter | None = None,
    ) -> None:
        """Initialize the exploratory.

        Args:
            dataset_config: Dataset configuration
            exploratory_config: Inspection configuration
            loader: Dataset loader (defaults to standard loader)
            reporter: Report writer (defaults to standard reporter)
        """
        self._dataset_config = dataset_config
        self._schema = self.get_schema()
        self._exploratory_config = exploratory_config or ExploratoryConfig()
        self._loader = loader or DatasetLoader(dataset_config, self._schema)
        self._reporter = reporter or ReportWritter(dataset_config)
        self._reporter.set_path_info_from_module()

    @abstractmethod
    def get_schema(self) -> Any:  # noqa: ANN401
        """Return the domain-specific schema.

        Returns:
            Schema object with target_column, timestamp_column, sender_column, etc.
        """
        raise NotImplementedError

    def run(self) -> dict[str, Any]:
        """Run comprehensive dataset exploratory.

        Returns:
            Dictionary containing all exploratory results
        """
        self._reporter.display_section("Dataset Exploratory")

        loaded = self._loader.load()
        df = loaded.frame

        # 1. Entity cardinality
        entity_cardinality = self._inspect_entity_cardinality(df)

        # 2. History distribution
        history_distribution = self._inspect_history_distribution(df)

        # 3. Fraud distribution on accounts with history
        fraud_distribution = self._inspect_fraud_distribution(df)

        # 4. Fraud position (first vs subsequent)
        fraud_position = self._inspect_fraud_position(df)

        # 5. History before fraud
        history_before_fraud = self._inspect_history_before_fraud(df)

        # 6. Sequence length availability
        sequence_length = self._inspect_sequence_length(df)

        # 7. Temporal distribution
        temporal_distribution = self._inspect_temporal_distribution(df)

        # Report all findings
        self._report_entity_cardinality(entity_cardinality)
        self._report_history_distribution(history_distribution)
        self._report_fraud_distribution(fraud_distribution)
        self._report_fraud_position(fraud_position)
        self._report_history_before_fraud(history_before_fraud)
        self._report_sequence_length(sequence_length)
        self._report_temporal_distribution(temporal_distribution)

        # Build summary
        summary = {
            "entity_cardinality": entity_cardinality,
            "history_distribution": history_distribution,
            "fraud_distribution": fraud_distribution,
            "fraud_position": fraud_position,
            "history_before_fraud": history_before_fraud,
            "sequence_length": sequence_length,
            "temporal_distribution": temporal_distribution,
        }
        self._reporter.update_summary(summary)

        # Save reports
        self._reporter.save_reports(
            text_filename="dataset_exploratory.txt",
            json_filename="dataset_exploratory.json",
        )

        return summary

    def _inspect_entity_cardinality(self, df: pd.DataFrame) -> dict[str, Any]:
        """How many unique customer/accounts?

        Returns:
            Dictionary with unique account counts and statistics
        """
        account_col = self._schema.sender_column
        unique_accounts = df[account_col].nunique()

        return {
            "n_unique_accounts": int(unique_accounts),
            "n_total_transactions": len(df),
            "avg_transactions_per_account": float(len(df) / unique_accounts),
        }

    def _inspect_history_distribution(self, df: pd.DataFrame) -> dict[str, Any]:
        """How many transactions per account?

        Returns:
            Dictionary with transaction count statistics per account
        """
        account_col = self._schema.sender_column
        counts = df.groupby(account_col).size()

        return {
            "min_transactions": int(counts.min()),
            "max_transactions": int(counts.max()),
            "median_transactions": float(counts.median()),
            "mean_transactions": float(counts.mean()),
            "p25_transactions": float(counts.quantile(0.25)),
            "p75_transactions": float(counts.quantile(0.75)),
            "p95_transactions": float(counts.quantile(0.95)),
            "p99_transactions": float(counts.quantile(0.99)),
        }

    def _inspect_fraud_distribution(self, df: pd.DataFrame) -> dict[str, Any]:
        """How many frauds occur on accounts with history?

        Returns:
            Dictionary with fraud distribution statistics
        """
        account_col = self._schema.sender_column
        target_col = self._schema.target_column

        # Count transactions per account
        account_counts = df.groupby(account_col).size()

        # Identify accounts with fraud
        fraud_by_account = df[df[target_col] == 1].groupby(account_col).size()

        # Total fraud transactions
        total_fraud = int(df[target_col].sum())

        # Accounts with fraud
        accounts_with_fraud = len(fraud_by_account)

        # Fraud on accounts with different history lengths
        fraud_mask = df[target_col] == 1
        fraud_history = df.loc[fraud_mask, account_col].map(account_counts)

        fraud_by_history = {}
        for window in self._exploratory_config.window_lengths:
            fraud_with_history = fraud_history >= window
            fraud_by_history[f"fraud_with_history_{window}"] = int(fraud_with_history.sum())
            fraud_by_history[f"pct_fraud_with_history_{window}"] = round(
                fraud_with_history.sum() / total_fraud * 100, 2
            ) if total_fraud > 0 else 0.0

        return {
            "total_fraud_transactions": total_fraud,
            "accounts_with_fraud": accounts_with_fraud,
            "accounts_with_history": int((account_counts >= 2).sum()),
            "accounts_with_history_and_fraud": int(
                (account_counts[account_counts >= 2].index.isin(fraud_by_account.index)).sum()
            ),
            **fraud_by_history,
        }

    def _inspect_fraud_position(self, df: pd.DataFrame) -> dict[str, Any]:
        """Does fraud appear more often as first or subsequent transaction?

        Returns:
            Dictionary with fraud position statistics
        """
        account_col = self._schema.sender_column
        target_col = self._schema.target_column
        timestamp_col = self._schema.timestamp_column

        # Sort by account and timestamp
        df_sorted = df.sort_values([account_col, timestamp_col])
        transaction_position = df_sorted.groupby(account_col).cumcount() + 1

        # Analyze fraud positions
        fraud_rows = transaction_position.loc[df_sorted[target_col] == 1]

        if len(fraud_rows) == 0:
            return {
                "n_fraud": 0,
                "n_fraud_first_transaction": 0,
                "n_fraud_second_transaction": 0,
                "n_fraud_third_transaction": 0,
                "n_fraud_after_third": 0,
                "pct_fraud_first_transaction": 0.0,
                "pct_fraud_second_transaction": 0.0,
                "pct_fraud_third_transaction": 0.0,
                "pct_fraud_after_third": 0.0,
                "median_position": 0.0,
                "mean_position": 0.0,
            }

        # Categorize positions
        first = (fraud_rows == 1).sum()
        second = (fraud_rows == 2).sum()
        third = (fraud_rows == 3).sum()
        after_third = (fraud_rows > 3).sum()

        total_fraud = len(fraud_rows)

        return {
            "n_fraud": total_fraud,
            "n_fraud_first_transaction": int(first),
            "n_fraud_second_transaction": int(second),
            "n_fraud_third_transaction": int(third),
            "n_fraud_after_third": int(after_third),
            "pct_fraud_first_transaction": round(first / total_fraud * 100, 2) if total_fraud > 0 else 0.0,
            "pct_fraud_second_transaction": round(second / total_fraud * 100, 2) if total_fraud > 0 else 0.0,
            "pct_fraud_third_transaction": round(third / total_fraud * 100, 2) if total_fraud > 0 else 0.0,
            "pct_fraud_after_third": round(after_third / total_fraud * 100, 2) if total_fraud > 0 else 0.0,
            "median_position": float(fraud_rows.median()) if total_fraud > 0 else 0.0,
            "mean_position": float(fraud_rows.mean()) if total_fraud > 0 else 0.0,
        }

    def _inspect_history_before_fraud(self, df: pd.DataFrame) -> dict[str, Any]:
        """How many normal transactions before fraud?

        Returns:
            Dictionary with history before fraud statistics
        """
        account_col = self._schema.sender_column
        target_col = self._schema.target_column
        timestamp_col = self._schema.timestamp_column

        # Sort by account and timestamp
        df_sorted = df.sort_values([account_col, timestamp_col])

        normal_mask = df_sorted[target_col].eq(0)
        normal_count = normal_mask.groupby(df_sorted[account_col], sort=False).cumsum() - normal_mask.astype(int)
        normal_before = normal_count.loc[df_sorted[target_col] == 1]
        n_fraud = len(normal_before)
        if n_fraud == 0:
            return {
                "n_fraud": 0,
                "median_normal_before_fraud": 0.0,
                "mean_normal_before_fraud": 0.0,
                "p25_normal_before_fraud": 0.0,
                "p75_normal_before_fraud": 0.0,
                "p95_normal_before_fraud": 0.0,
                "pct_fraud_with_no_history": 0.0,
                "pct_fraud_with_1_4_history": 0.0,
                "pct_fraud_with_5_9_history": 0.0,
                "pct_fraud_with_10_plus_history": 0.0,
            }

        return {
            "n_fraud": n_fraud,
            "median_normal_before_fraud": float(normal_before.median()),
            "mean_normal_before_fraud": float(normal_before.mean()),
            "p25_normal_before_fraud": float(normal_before.quantile(0.25)),
            "p75_normal_before_fraud": float(normal_before.quantile(0.75)),
            "p95_normal_before_fraud": float(normal_before.quantile(0.95)),
            "pct_fraud_with_no_history": round((normal_before == 0).sum() / n_fraud * 100, 2),
            "pct_fraud_with_1_4_history": round(((normal_before >= 1) & (normal_before <= 4)).sum() / n_fraud * 100, 2), # noqa: E501
            "pct_fraud_with_5_9_history": round(((normal_before >= 5) & (normal_before <= 9)).sum() / n_fraud * 100, 2), # noqa: E501
            "pct_fraud_with_10_plus_history": round((normal_before >= 10).sum() / n_fraud * 100, 2),
        }

    def _inspect_sequence_length(self, df: pd.DataFrame) -> dict[str, Any]:
        """Are there enough sequences with length 3, 5, 10, etc?

        Returns:
            Dictionary with sequence length availability statistics
        """
        account_col = self._schema.sender_column
        account_counts = df.groupby(account_col).size()

        # Count accounts with sufficient history for each window length
        sequence_counts = {}
        for window in self._exploratory_config.window_lengths:
            n_accounts = int((account_counts >= window).sum())
            sequence_counts[f"accounts_with_sequence_{window}"] = n_accounts
            sequence_counts[f"pct_accounts_with_sequence_{window}"] = round(
                n_accounts / len(account_counts) * 100, 2
            )

        # Total possible sequences (each account can have multiple sequences)
        total_sequences = {}
        for window in self._exploratory_config.window_lengths:
            n_sequences = int((account_counts - window + 1).clip(lower=0).sum())
            total_sequences[f"total_sequences_{window}"] = n_sequences

        return {
            "n_accounts": len(account_counts),
            **sequence_counts,
            **total_sequences,
        }

    def _inspect_temporal_distribution(self, df: pd.DataFrame) -> dict[str, Any]:
        """Is fraud distributed throughout the dataset period?

        Returns:
            Dictionary with temporal distribution statistics
        """
        target_col = self._schema.target_column
        timestamp_col = self._schema.timestamp_column

        raw_timestamps = df[timestamp_col]
        if pd.api.types.is_numeric_dtype(raw_timestamps):
            unit = getattr(self._schema, "timestamp_unit", "h")
            timestamps = pd.to_timedelta(raw_timestamps, unit=unit, errors="coerce")
            elapsed_days = timestamps.dt.total_seconds() / 86400
            time_range = timestamps.max() - timestamps.min()
        else:
            timestamps = pd.to_datetime(raw_timestamps, errors="coerce")
            elapsed_days = timestamps.astype("int64", copy=False) / 86400e9
            elapsed_days = elapsed_days.where(timestamps.notna())
            time_range = timestamps.max() - timestamps.min()

        range_days = float(time_range.total_seconds() / 86400) if pd.notna(time_range) else 0.0
        if range_days > 30:
            periods = np.floor(elapsed_days / 30).astype("Int64")
            period_label = "30-day"
        elif range_days > 7:
            periods = np.floor(elapsed_days / 7).astype("Int64")
            period_label = "week"
        elif range_days >= 1:
            periods = np.floor(elapsed_days).astype("Int64")
            period_label = "day"
        else:
            periods = np.floor(elapsed_days * 24).astype("Int64")
            period_label = "hour"

        # Calculate fraud rate per period
        period_stats = df.groupby(periods)[target_col].agg(
            n_transactions="size",
            n_fraud="sum",
            fraud_rate="mean"
        )

        # Calculate temporal drift
        fraud_rates = period_stats["fraud_rate"]
        if len(fraud_rates) >= 2:
            min_rate = fraud_rates.min()
            max_rate = fraud_rates.max()
            temporal_drift = max_rate / min_rate if min_rate > 0 else 999.99
        else:
            temporal_drift = 0.0

        # Convert period_stats to serializable format
        period_stats_dict = {}
        for period, row in period_stats.iterrows():
            period_key = str(period)
            period_stats_dict[period_key] = {
                "n_transactions": int(row["n_transactions"]),
                "n_fraud": int(row["n_fraud"]),
                "fraud_rate": float(row["fraud_rate"]),
            }

        return {
            "time_start": str(raw_timestamps.min()),
            "time_end": str(raw_timestamps.max()),
            "time_range_days": range_days,
            "n_periods": len(period_stats),
            "period_type": period_label,
            "min_fraud_rate_per_period": float(fraud_rates.min()) if len(fraud_rates) > 0 else 0.0,
            "max_fraud_rate_per_period": float(fraud_rates.max()) if len(fraud_rates) > 0 else 0.0,
            "mean_fraud_rate_per_period": float(fraud_rates.mean()) if len(fraud_rates) > 0 else 0.0,
            "temporal_drift_ratio": round(temporal_drift, 2),
            "period_stats": period_stats_dict,
        }

    def _report_entity_cardinality(self, data: dict[str, Any]) -> None:
        self._reporter.display_section("1. Entity Cardinality")
        self._reporter.display_text(
            f"Unique accounts: {data['n_unique_accounts']:,} | "
            f"Total transactions: {data['n_total_transactions']:,} | "
            f"Avg transactions/account: {data['avg_transactions_per_account']:.2f}"
        )

    def _report_history_distribution(self, data: dict[str, Any]) -> None:
        self._reporter.display_section("2. History Distribution (Transactions per Account)")
        table_data = {
            "Statistic": ["Min", "Max", "Median", "Mean", "P25", "P75", "P95", "P99"],
            "Value": [
                data["min_transactions"],
                data["max_transactions"],
                data["median_transactions"],
                round(data["mean_transactions"], 2),
                data["p25_transactions"],
                data["p75_transactions"],
                data["p95_transactions"],
                data["p99_transactions"],
            ],
        }
        self._reporter.display_table(pd.DataFrame(table_data).set_index("Statistic"))

    def _report_fraud_distribution(self, data: dict[str, Any]) -> None:
        self._reporter.display_section("3. Fraud Distribution")
        self._reporter.display_text(
            f"Total fraud transactions: {data['total_fraud_transactions']:,} | "
            f"Accounts with fraud: {data['accounts_with_fraud']:,} | "
            f"Accounts with history (>=2 txns): {data['accounts_with_history']:,}"
        )
        self._reporter.display_text(
            f"Accounts with history AND fraud: {data['accounts_with_history_and_fraud']:,}"
        )

        # Report fraud by history length
        window_data = []
        for window in self._exploratory_config.window_lengths:
            window_data.append({
                "Window": window,
                "Fraud with history": data[f"fraud_with_history_{window}"],
                "Pct of fraud": data[f"pct_fraud_with_history_{window}"],
            })
        self._reporter.display_section("Fraud by Account History Length")
        self._reporter.display_table(pd.DataFrame(window_data).set_index("Window"))

    def _report_fraud_position(self, data: dict[str, Any]) -> None:
        self._reporter.display_section("4. Fraud Position (First vs Subsequent Transaction)")
        self._reporter.display_text(
            f"Total fraud: {data['n_fraud']:,} | "
            f"Median position: {data.get('median_position', 0):.1f} | "
            f"Mean position: {data.get('mean_position', 0):.1f}"
        )

        position_data = {
            "Position": ["1st", "2nd", "3rd", "After 3rd"],
            "Count": [
                data["n_fraud_first_transaction"],
                data["n_fraud_second_transaction"],
                data["n_fraud_third_transaction"],
                data["n_fraud_after_third"],
            ],
            "Percentage": [
                data["pct_fraud_first_transaction"],
                data["pct_fraud_second_transaction"],
                data["pct_fraud_third_transaction"],
                data["pct_fraud_after_third"],
            ],
        }
        self._reporter.display_table(pd.DataFrame(position_data).set_index("Position"))

    def _report_history_before_fraud(self, data: dict[str, Any]) -> None:
        self._reporter.display_section("5. History Before Fraud (Normal Transactions)")
        self._reporter.display_text(
            f"Total fraud: {data['n_fraud']:,} | "
            f"Median normal before: {data['median_normal_before_fraud']:.1f} | "
            f"Mean normal before: {data['mean_normal_before_fraud']:.1f}"
        )

        history_data = {
            "Statistic": ["Median", "Mean", "P25", "P75", "P95"],
            "Value": [
                data["median_normal_before_fraud"],
                round(data["mean_normal_before_fraud"], 2),
                data["p25_normal_before_fraud"],
                data["p75_normal_before_fraud"],
                data["p95_normal_before_fraud"],
            ],
        }
        self._reporter.display_table(pd.DataFrame(history_data).set_index("Statistic"))

        distribution_data = {
            "History Range": ["0 (no history)", "1-4", "5-9", "10+"],
            "Percentage of Fraud": [
                data["pct_fraud_with_no_history"],
                data["pct_fraud_with_1_4_history"],
                data["pct_fraud_with_5_9_history"],
                data["pct_fraud_with_10_plus_history"],
            ],
        }
        self._reporter.display_section("Distribution of Normal Transactions Before Fraud")
        self._reporter.display_table(pd.DataFrame(distribution_data).set_index("History Range"))

    def _report_sequence_length(self, data: dict[str, Any]) -> None:
        self._reporter.display_section("6. Sequence Length Availability")
        self._reporter.display_text(f"Total accounts: {data['n_accounts']:,}")

        # Account coverage by window
        account_data = []
        for window in self._exploratory_config.window_lengths:
            account_data.append({
                "Window": window,
                "Accounts": data[f"accounts_with_sequence_{window}"],
                "Pct Accounts": data[f"pct_accounts_with_sequence_{window}"],
            })
        self._reporter.display_section("Accounts with Sufficient History")
        self._reporter.display_table(pd.DataFrame(account_data).set_index("Window"))

        # Total sequences by window
        sequence_data = []
        for window in self._exploratory_config.window_lengths:
            sequence_data.append({
                "Window": window,
                "Total Sequences": data[f"total_sequences_{window}"],
            })
        self._reporter.display_section("Total Possible Sequences")
        self._reporter.display_table(pd.DataFrame(sequence_data).set_index("Window"))

    def _report_temporal_distribution(self, data: dict[str, Any]) -> None:
        self._reporter.display_section("7. Temporal Distribution")
        self._reporter.display_text(
            f"Time range: {data['time_start']} to {data['time_end']} "
            f"({data['time_range_days']:.1f} days)"
        )
        self._reporter.display_text(
            f"Periods: {data['n_periods']} ({data['period_type']}) | "
            f"Min fraud rate: {data['min_fraud_rate_per_period']:.4f} | "
            f"Max fraud rate: {data['max_fraud_rate_per_period']:.4f}"
        )
        self._reporter.display_text(
            f"Mean fraud rate: {data['mean_fraud_rate_per_period']:.4f} | "
            f"Temporal drift ratio: {data['temporal_drift_ratio']:.2f}x"
        )

        if data["temporal_drift_ratio"] > 2.0:
            self._reporter.display_warning(
                f"High temporal drift detected ({data['temporal_drift_ratio']:.2f}x). "
                "Fraud rate varies significantly across time periods."
            )
