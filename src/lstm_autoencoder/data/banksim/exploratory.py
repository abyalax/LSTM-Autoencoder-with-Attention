"""Exploration of the BankSim datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.banksim.schema import BankSimSchema
from lstm_autoencoder.data.config import ExploratoryConfig
from lstm_autoencoder.data.exploratory import DatasetExploratory
from lstm_autoencoder.data.loader import DatasetLoader
from lstm_autoencoder.evaluation.config import SplitConfig
from lstm_autoencoder.evaluation.paysim_synthetic_financial.metrics import (
    SIGNAL_THRESHOLD,
    SignalCheckResult,
    UnivariateSignalChecker,
)
from lstm_autoencoder.reporting.reporter import ReportWritter

TARGET_COLUMN = "fraud"
ACCOUNT_COLUMN = "customer"
STEP_COLUMN = "step"
TYPE_COLUMN = "category"
WINDOW_LENGTHS: tuple[int, ...] = (1, 3, 5, 10)
HISTORY_STRATA_BINS: tuple[int, ...] = (0, 1, 4, 9, np.inf)
HISTORY_STRATA_LABELS: tuple[str, ...] = ("1", "2-4", "5-9", ">=10")


class BankSimDatasetExploratory(DatasetExploratory):
    """Exploratory for BankSim datasets.

    Inherits from DatasetExploratory and adds BankSim-specific functional exploration
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
        super().__init__(dataset_config, loader=loader, reporter=reporter)
        self._exploratory_config = exploratory_config or ExploratoryConfig()
        self._split_config = split_config or SplitConfig()
        # Override loader methods for BankSim-specific handling
        self._loader._parse_timestamps = self._skip_timestamp_parsing
        self._loader._normalize_target = self._skip_target_normalization

    def get_schema(self) -> BankSimSchema:
        """Return the BankSim schema.

        Returns:
            BankSimSchema with BankSim-specific column names and settings
        """
        return BankSimSchema()

    @staticmethod
    def _skip_timestamp_parsing(series: pd.Series) -> pd.Series:
        """Skip timestamp parsing for BankSim - step is already an integer."""
        return series

    @staticmethod
    def _skip_target_normalization(series: pd.Series) -> pd.Series:
        """Skip target normalization for BankSim - fraud is already 0/1."""
        return series

    def run_functional(self) -> dict[str, object]:
        """Run structural and leakage-aware signal checks before committing to BankSim."""
        self._reporter.display_section("BankSim Datasets Exploration")

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
        """Compute fraud prevalence overall and by transaction type."""
        n_rows = len(df)
        n_fraud = int(df[TARGET_COLUMN].sum())
        by_type = None
        if TYPE_COLUMN in df.columns:
            by_type = (
                df.groupby(TYPE_COLUMN)[TARGET_COLUMN]
                .agg(n="count", n_fraud="sum")
                .assign(fraud_rate=lambda t: t["n_fraud"] / t["n"])
            )
        return {
            "n_rows": n_rows,
            "n_fraud": n_fraud,
            "prevalence": n_fraud / n_rows,
            "by_type": by_type,
        }

    def _inspect_account_viability(self, df: pd.DataFrame) -> dict[str, object]:
        """Check how many customer accounts support each candidate window length."""
        counts = df.groupby(ACCOUNT_COLUMN).size()
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
        """Check window viability and what fraction of fraud transactions actually have sufficient customer history."""
        customer_counts = df.groupby(ACCOUNT_COLUMN).size()

        df = df.assign(customer_history_len=df[ACCOUNT_COLUMN].map(customer_counts))
        fraud_rows = df[df[TARGET_COLUMN] == 1]

        coverage = {
            window: round(float((fraud_rows["customer_history_len"] >= window).mean()) * 100, 2)
            for window in WINDOW_LENGTHS
        }
        return {
            "n_relevant_rows": len(df),
            "n_relevant_customers": int(customer_counts.shape[0]),
            "customer_history_median": float(customer_counts.median()),
            "customer_history_p95": float(customer_counts.quantile(0.95)),
            "pct_fraud_with_window": coverage,
        }

    def _inspect_fraud_history_strata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Break down fraud transactions by customer history-length bucket, checking
        whether thin-history fraud concentrates in particular amount or time ranges.
        """
        customer_counts = df.groupby(ACCOUNT_COLUMN).size()

        fraud_rows = df[df[TARGET_COLUMN] == 1].assign(history_len=lambda t: t[ACCOUNT_COLUMN].map(customer_counts))
        fraud_rows = fraud_rows.assign(
            history_bucket=pd.cut(
                fraud_rows["history_len"],
                bins=HISTORY_STRATA_BINS,
                labels=HISTORY_STRATA_LABELS,
            )
        )

        strata = fraud_rows.groupby("history_bucket", observed=True).agg(
            n_fraud=(TARGET_COLUMN, "size"),
            amount_median=("amount", "median"),
            amount_p95=("amount", lambda s: s.quantile(0.95)),
            step_min=(STEP_COLUMN, "min"),
            step_max=(STEP_COLUMN, "max"),
        )
        strata["pct_of_fraud"] = (strata["n_fraud"] / len(fraud_rows) * 100).round(2)
        return strata

    def _compare_history_bucket_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare customer history-length bucket distribution between fraud and
        non-fraud rows, to check whether thin history is fraud-specific or a
        general trait of the fraud-relevant transaction population.
        """
        customer_counts = df.groupby(ACCOUNT_COLUMN).size()
        df = df.assign(history_len=df[ACCOUNT_COLUMN].map(customer_counts))
        df = df.assign(
            history_bucket=pd.cut(
                df["history_len"],
                bins=HISTORY_STRATA_BINS,
                labels=HISTORY_STRATA_LABELS,
            )
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
        Simulate the 70/15/15 customer-level split; report fraud rate per split.
        Split by customer so no customer appears in multiple splits.
        """
        from lstm_autoencoder.evaluation.config import SplitConfig
        from lstm_autoencoder.evaluation.receiver_split import ReceiverSplitter

        split_config = SplitConfig(train_fraction=ratios[0], validation_fraction=ratios[1])
        from lstm_autoencoder.data.banksim.causal_features import account_identity

        account = account_identity(df)
        groups = ReceiverSplitter(split_config, random_seed=42).split(account)

        parts = {name: df[account.isin(ids)] for name, ids in groups.items()}

        return {
            name: {
                "n": len(part),
                "n_customers": int(account.loc[part.index].nunique()),
                "fraud_rate": float(part[TARGET_COLUMN].mean()),
            }
            for name, part in parts.items()
        }

    def _report_history_bucket_comparison(self, comparison: pd.DataFrame) -> None:
        self._reporter.display_section("History bucket: fraud vs normal distribution (lift)")
        self._reporter.display_table(comparison)

    def _report_fraud_history_strata(self, strata: pd.DataFrame) -> None:
        self._reporter.display_section("Fraud by customer history-length bucket")
        self._reporter.display_table(strata)

    def _report_fraud_window_coverage(self, coverage: dict[str, object]) -> None:
        self._reporter.display_text(
            f"Rows: {coverage['n_relevant_rows']:,} | "
            f"unique customers: {coverage['n_relevant_customers']:,} | "
            f"customer history median: {coverage['customer_history_median']} | "
            f"p95: {coverage['customer_history_p95']}"
        )
        self._reporter.display_section("Fraud transactions with customer window >= W (%)")
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
            self._reporter.display_section("Fraud rate by transaction type")
            self._reporter.display_table(target["by_type"])

    def _report_account_viability(self, viability: dict[str, object]) -> None:
        self._reporter.display_text(
            f"Customer accounts: {viability['n_accounts']:,} | "
            f"txn/account median: {viability['txn_per_account_median']} | "
            f"p95: {viability['txn_per_account_p95']}"
        )
        self._reporter.display_section("Accounts with window >= W (%)")
        self._reporter.display_table(
            pd.DataFrame.from_dict(viability["pct_accounts_with_window"], orient="index", columns=["pct_accounts"])
        )

    def _report_split(self, split: dict[str, dict[str, float]]) -> None:
        self._reporter.display_section("Stratified split by customer (fraud rate)")
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
