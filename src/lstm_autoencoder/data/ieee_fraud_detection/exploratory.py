"""Exploration of the IEEE-CIS Fraud Detection datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.exploratory import DatasetExploratory, ExploratoryConfig
from lstm_autoencoder.data.ieee_fraud_detection.schema import IEEEFraudDetectionSchema
from lstm_autoencoder.data.loader import DatasetLoader
from lstm_autoencoder.evaluation.config import SplitConfig
from lstm_autoencoder.evaluation.paysim_synthetic_financial.metrics import (
    SIGNAL_THRESHOLD,
    SignalCheckResult,
    UnivariateSignalChecker,
)
from lstm_autoencoder.reporting.reporter import ReportWritter

TARGET_COLUMN = "isFraud"
ACCOUNT_COLUMN = "card1"  # Fallback, actual uid is composite in causal_features
TIMESTAMP_COLUMN = "TransactionDT"
TYPE_COLUMN = "ProductCD"
WINDOW_LENGTHS: tuple[int, ...] = (1, 3, 5, 10)
HISTORY_STRATA_BINS: tuple[int, ...] = (0, 1, 4, 9, np.inf)
HISTORY_STRATA_LABELS: tuple[str, ...] = ("1", "2-4", "5-9", ">=10")


class IEEEFraudDetectionDatasetExploratory(DatasetExploratory):
    """Exploratory for IEEE-CIS Fraud Detection datasets.

    Inherits from DatasetExploratory and adds IEEE-specific functional exploration
    including structural analysis, leakage-aware signal checks, and split simulation.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        exploratory_config: ExploratoryConfig | None = None,
        split_config: SplitConfig | None = None,
        loader: DatasetLoader | None = None,
        reporter: ReportWritter | None = None,
    ) -> None:
        """Create an explorer; the loader and reporter default to standard ones."""
        super().__init__(dataset_config, exploratory_config=exploratory_config, loader=loader, reporter=reporter)
        self._split_config = split_config or SplitConfig()
        # Override loader methods for IEEE-specific handling
        self._loader._parse_timestamps = self._skip_timestamp_parsing
        self._loader._normalize_target = self._skip_target_normalization

    def _inspect_entity_cardinality(self, df: pd.DataFrame) -> dict[str, object]:
        """How many unique customer/accounts using composite uid."""
        from lstm_autoencoder.data.ieee_fraud_detection.causal_features import account_identity

        account = account_identity(df)
        unique_accounts = account.nunique()

        return {
            "n_unique_accounts": int(unique_accounts),
            "n_total_transactions": len(df),
            "avg_transactions_per_account": float(len(df) / unique_accounts),
        }

    def _inspect_history_distribution(self, df: pd.DataFrame) -> dict[str, object]:
        """How many transactions per account using composite uid."""
        from lstm_autoencoder.data.ieee_fraud_detection.causal_features import account_identity

        account = account_identity(df)
        counts = account.value_counts()

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

    def _inspect_fraud_distribution(self, df: pd.DataFrame) -> dict[str, object]:
        """How many frauds occur on accounts with history using composite uid."""
        from lstm_autoencoder.data.ieee_fraud_detection.causal_features import account_identity

        account = account_identity(df)
        target_col = self._schema.target_column

        # Count transactions per account
        account_counts = account.value_counts()

        # Identify accounts with fraud
        fraud_mask = df[target_col] == 1
        fraud_by_account = account[fraud_mask].value_counts()

        # Total fraud transactions
        total_fraud = int(df[target_col].sum())

        # Accounts with fraud
        accounts_with_fraud = len(fraud_by_account)

        # Fraud on accounts with different history lengths
        fraud_history = account[fraud_mask].map(account_counts)

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

    def _inspect_fraud_position(self, df: pd.DataFrame) -> dict[str, object]:
        """Does fraud appear more often as first or subsequent transaction using composite uid."""
        from lstm_autoencoder.data.ieee_fraud_detection.causal_features import account_identity

        account = account_identity(df)
        target_col = self._schema.target_column
        timestamp_col = self._schema.timestamp_column

        # Sort by account and timestamp
        df_sorted = df.copy()
        df_sorted["account"] = account
        df_sorted = df_sorted.sort_values(["account", timestamp_col])
        transaction_position = df_sorted.groupby("account").cumcount() + 1

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

    def _inspect_history_before_fraud(self, df: pd.DataFrame) -> dict[str, object]:
        """How many normal transactions before fraud using composite uid."""
        from lstm_autoencoder.data.ieee_fraud_detection.causal_features import account_identity

        account = account_identity(df)
        target_col = self._schema.target_column
        timestamp_col = self._schema.timestamp_column

        # Sort by account and timestamp
        df_sorted = df.copy()
        df_sorted["account"] = account
        df_sorted = df_sorted.sort_values(["account", timestamp_col])

        normal_mask = df_sorted[target_col].eq(0)
        normal_count = normal_mask.groupby(df_sorted["account"], sort=False).cumsum() - normal_mask.astype(int)
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
            "pct_fraud_with_1_4_history": round(((normal_before >= 1) & (normal_before <= 4)).sum() / n_fraud * 100, 2),
            "pct_fraud_with_5_9_history": round(((normal_before >= 5) & (normal_before <= 9)).sum() / n_fraud * 100, 2),
            "pct_fraud_with_10_plus_history": round((normal_before >= 10).sum() / n_fraud * 100, 2),
        }

    def _inspect_sequence_length(self, df: pd.DataFrame) -> dict[str, object]:
        """Are there enough sequences with length 3, 5, 10, etc using composite uid."""
        from lstm_autoencoder.data.ieee_fraud_detection.causal_features import account_identity

        account = account_identity(df)
        account_counts = account.value_counts()

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

    def get_schema(self) -> IEEEFraudDetectionSchema:
        """Return the IEEE-CIS Fraud Detection schema.

        Returns:
            IEEEFraudDetectionSchema with IEEE-specific column names and settings
        """
        return IEEEFraudDetectionSchema()

    @staticmethod
    def _skip_timestamp_parsing(series: pd.Series) -> pd.Series:
        """Skip timestamp parsing for IEEE - TransactionDT is already an integer (seconds)."""
        return series

    @staticmethod
    def _skip_target_normalization(series: pd.Series) -> pd.Series:
        """Skip target normalization for IEEE - isFraud is already 0/1."""
        return series

    def run_functional(self) -> dict[str, object]:
        """Run structural and leakage-aware signal checks before committing to IEEE."""
        self._reporter.display_section("IEEE-CIS Fraud Detection Datasets Exploration")

        loaded = self._loader.load()
        df = loaded.frame

        structure = self._inspect_structure(df)
        target = self._inspect_target(df)
        account_viability = self._inspect_account_viability(df)
        fraud_window_coverage = self._inspect_fraud_window_coverage(df)
        fraud_history_strata = self._inspect_fraud_history_strata(df)
        history_bucket_comparison = self._compare_history_bucket_distribution(df)

        split = self._simulate_chronological_split(df)
        checker = UnivariateSignalChecker()
        signal_without_leakage = checker.check_univariate_signal(df, target_column=TARGET_COLUMN, exclude_leakage=True)

        self._report_structure(structure)
        self._report_target(target)
        self._report_account_viability(account_viability)
        self._report_split(split)
        self._report_signal(signal_without_leakage)
        self._report_fraud_window_coverage(fraud_window_coverage)
        self._report_fraud_history_strata(fraud_history_strata)
        self._report_history_bucket_comparison(history_bucket_comparison)

        # Build summary
        summary = {
            "structure": structure,
            "target": target,
            "account_viability": account_viability,
            "fraud_window_coverage": fraud_window_coverage,
            "fraud_history_strata": fraud_history_strata.to_dict(),
            "history_bucket_comparison": history_bucket_comparison.to_dict(),
            "split": split,
            "signal_without_leakage": [r.__dict__ for r in signal_without_leakage],
        }
        self._reporter.update_summary(summary)

        # Save reports
        self._reporter.save_reports(
            text_filename="dataset_report.txt",
            json_filename="dataset_report.json",
        )

        return summary

    def _inspect_structure(self, df: pd.DataFrame) -> dict[str, object]:
        """Collect shape, dtypes, missing values, and duplicate counts."""
        n_rows, n_cols = df.shape
        missing = df.isna().sum()
        return {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "n_missing": missing.to_dict(),
            "pct_missing": (missing / n_rows * 100).round(3).to_dict(),
            "n_unique": df.nunique().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
        }

    def _inspect_target(self, df: pd.DataFrame) -> dict[str, object]:
        """Compute fraud prevalence overall and by product code."""
        n_rows = len(df)
        n_fraud = int(df[TARGET_COLUMN].sum())
        by_type = None
        if TYPE_COLUMN in df.columns:
            by_type = (
                df.groupby(TYPE_COLUMN)[TARGET_COLUMN]
                .agg(n="count", n_fraud="sum")
            )
            by_type["fraud_rate"] = by_type["n_fraud"] / by_type["n"]
        return {
            "n_rows": n_rows,
            "n_fraud": n_fraud,
            "prevalence": n_fraud / n_rows,
            "by_type": by_type,
        }

    def _inspect_account_viability(self, df: pd.DataFrame) -> dict[str, object]:
        """Check how many card accounts support each candidate window length."""
        from lstm_autoencoder.data.ieee_fraud_detection.causal_features import account_identity

        account = account_identity(df)
        counts = account.value_counts()
        n_accounts = int(counts.shape[0])
        pct_with_window = {
            window: round(int((counts >= window).sum()) / n_accounts * 100, 2) for window in WINDOW_LENGTHS
        }
        return {
            "n_accounts": n_accounts,
            "txn_per_account_median": float(counts.median()),
            "txn_per_account_p95": float(counts.quantile(0.95)),
            "pct_accounts_with_window": pct_with_window,
        }

    def _inspect_fraud_window_coverage(self, df: pd.DataFrame) -> dict[str, object]:
        """Check window viability and what fraction of fraud transactions actually have sufficient card history."""
        from lstm_autoencoder.data.ieee_fraud_detection.causal_features import account_identity

        account = account_identity(df)
        card_counts = account.value_counts()

        df = df.copy()
        df["card_history_len"] = df.index.map(card_counts)
        fraud_rows = df[df[TARGET_COLUMN] == 1]

        coverage = {
            window: round(float((fraud_rows["card_history_len"] >= window).mean()) * 100, 2)
            for window in WINDOW_LENGTHS
        }
        return {
            "n_relevant_rows": len(df),
            "n_relevant_cards": int(card_counts.shape[0]),
            "card_history_median": float(card_counts.median()),
            "card_history_p95": float(card_counts.quantile(0.95)),
            "pct_fraud_with_window": coverage,
        }

    def _inspect_fraud_history_strata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Break down fraud transactions by card history-length bucket, checking
        whether thin-history fraud concentrates in particular amount or time ranges.
        """
        from lstm_autoencoder.data.ieee_fraud_detection.causal_features import account_identity

        account = account_identity(df)
        card_counts = account.value_counts()

        fraud_rows = df[df[TARGET_COLUMN] == 1].copy()
        fraud_rows["history_len"] = fraud_rows.index.map(card_counts)
        fraud_rows["history_bucket"] = pd.cut(
            fraud_rows["history_len"],
            bins=HISTORY_STRATA_BINS,
            labels=HISTORY_STRATA_LABELS,
        )

        strata = fraud_rows.groupby("history_bucket", observed=True).agg(
            n_fraud=(TARGET_COLUMN, "size"),
            amount_median=("TransactionAmt", "median"),
            amount_p95=("TransactionAmt", lambda s: s.quantile(0.95)),
            timestamp_min=(TIMESTAMP_COLUMN, "min"),
            timestamp_max=(TIMESTAMP_COLUMN, "max"),
        )
        strata["pct_of_fraud"] = (strata["n_fraud"] / len(fraud_rows) * 100).round(2)
        return strata

    def _compare_history_bucket_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare card history-length bucket distribution between fraud and
        non-fraud rows, to check whether thin history is fraud-specific or a
        general trait of the fraud-relevant transaction population.
        """
        from lstm_autoencoder.data.ieee_fraud_detection.causal_features import account_identity

        account = account_identity(df)
        card_counts = account.value_counts()
        df = df.copy()
        df["history_len"] = df.index.map(card_counts)
        df["history_bucket"] = pd.cut(
            df["history_len"],
            bins=HISTORY_STRATA_BINS,
            labels=HISTORY_STRATA_LABELS,
        )

        counts = (
            df.groupby([TARGET_COLUMN, "history_bucket"], observed=True)
            .size()
            .rename("n")
            .reset_index()
            .pivot(index="history_bucket", columns=TARGET_COLUMN, values="n")
            .fillna(0)
        )
        counts.columns = ["normal", "fraud"]
        counts["pct_normal"] = (counts["normal"] / counts["normal"].sum() * 100).round(2)
        counts["pct_fraud"] = (counts["fraud"] / counts["fraud"].sum() * 100).round(2)
        counts["lift"] = (counts["pct_fraud"] / counts["pct_normal"]).round(2)
        return counts

    def _simulate_chronological_split(
        self,
        df: pd.DataFrame,
        ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> dict[str, dict[str, float]]:
        """
        Simulate the 70/15/15 card-level split; report fraud rate per split.
        Split by card so no card appears in multiple splits.
        """
        from lstm_autoencoder.evaluation.config import SplitConfig
        from lstm_autoencoder.evaluation.receiver_split import ReceiverSplitter
        from lstm_autoencoder.data.ieee_fraud_detection.causal_features import account_identity

        split_config = SplitConfig(train_fraction=ratios[0], validation_fraction=ratios[1])
        account = account_identity(df)
        groups = ReceiverSplitter(split_config, random_seed=42).split(account)

        parts = {name: df[account.isin(ids)] for name, ids in groups.items()}

        return {
            name: {
                "n": len(part),
                "n_cards": int(account.loc[part.index].nunique()),
                "fraud_rate": float(part[TARGET_COLUMN].mean()),
            }
            for name, part in parts.items()
        }

    def _report_history_bucket_comparison(self, comparison: pd.DataFrame) -> None:
        self._reporter.display_section("History bucket: fraud vs normal distribution (lift)")
        self._reporter.display_table(comparison)

    def _report_fraud_history_strata(self, strata: pd.DataFrame) -> None:
        self._reporter.display_section("Fraud by card history-length bucket")
        self._reporter.display_table(strata)

    def _report_fraud_window_coverage(self, coverage: dict[str, object]) -> None:
        self._reporter.display_text(
            f"Rows: {coverage['n_relevant_rows']:,} | "
            f"unique cards: {coverage['n_relevant_cards']:,} | "
            f"card history median: {coverage['card_history_median']} | "
            f"p95: {coverage['card_history_p95']}"
        )
        self._reporter.display_section("Fraud transactions with card window >= W (%)")
        self._reporter.display_table(
            pd.DataFrame.from_dict(coverage["pct_fraud_with_window"], orient="index", columns=["pct_fraud"])
        )

    def _report_structure(self, structure: dict[str, object]) -> None:
        self._reporter.display_text(
            f"Rows: {structure['n_rows']:,} | Columns: {structure['n_cols']} | "
            f"Duplicate rows: {structure['duplicate_rows']}"
        )
        self._reporter.display_section("Dtypes")
        self._reporter.display_table(pd.DataFrame.from_dict(structure["dtypes"], orient="index", columns=["dtype"]))
        self._reporter.display_section("Missing (%)")
        self._reporter.display_table(
            pd.DataFrame.from_dict(structure["pct_missing"], orient="index", columns=["missing_pct"])
        )

    def _report_target(self, target: dict[str, object]) -> None:
        self._reporter.display_text(f"Fraud: {target['n_fraud']:,} / {target['n_rows']:,} ({target['prevalence']:.4%})")
        if target["by_type"] is not None:
            self._reporter.display_section("Fraud rate by product code")
            self._reporter.display_table(target["by_type"])

    def _report_account_viability(self, viability: dict[str, object]) -> None:
        self._reporter.display_text(
            f"Card accounts: {viability['n_accounts']:,} | "
            f"txn/account median: {viability['txn_per_account_median']} | "
            f"p95: {viability['txn_per_account_p95']}"
        )
        self._reporter.display_section("Accounts with window >= W (%)")
        self._reporter.display_table(
            pd.DataFrame.from_dict(viability["pct_accounts_with_window"], orient="index", columns=["pct_accounts"])
        )

    def _report_split(self, split: dict[str, dict[str, float]]) -> None:
        self._reporter.display_section("Stratified split by card (fraud rate)")
        self._reporter.display_table(pd.DataFrame.from_dict(split, orient="index"))

    def _report_signal(self, without_leakage: list[SignalCheckResult]) -> None:
        self._reporter.display_section("Signal check: without leakage columns")
        self._reporter.display_section("Without leakage")
        self._reporter.display_table(
            pd.DataFrame.from_dict({r.feature: round(r.auc, 4) for r in without_leakage}, orient="index", columns=["auc"])
        )

        max_gap = max((abs(r.auc - 0.5) for r in without_leakage), default=0.0)
        if max_gap < SIGNAL_THRESHOLD:
            self._reporter.display_text(
                "[WARNING] No feature shows usable signal once leakage columns are "
                "excluded. Stop before building the full LSTM-AE pipeline — this is "
                "the same fatal pattern seen in the previous dataset."
            )
