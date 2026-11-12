"""Exploration of the PaySim synthetic financial datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.exploratory import DatasetExploratory, ExploratoryConfig
from lstm_autoencoder.data.loader import DatasetLoader
from lstm_autoencoder.data.paysim_synthetic_financial.schema import PaySimSchema
from lstm_autoencoder.evaluation.config import SplitConfig
from lstm_autoencoder.evaluation.paysim_synthetic_financial.metrics import (
    SIGNAL_THRESHOLD,
    SignalCheckResult,
    UnivariateSignalChecker,
)
from lstm_autoencoder.reporting.reporter import ReportWritter

TARGET_COLUMN = "isFraud"
ACCOUNT_COLUMN = "nameOrig"
STEP_COLUMN = "step"
TYPE_COLUMN = "type"
WINDOW_LENGTHS: tuple[int, ...] = (1, 3, 5, 10)
HISTORY_STRATA_BINS: tuple[int, ...] = (0, 1, 4, 9, np.inf)
HISTORY_STRATA_LABELS: tuple[str, ...] = ("1", "2-4", "5-9", ">=10")


class PaySimDatasetExploratory(DatasetExploratory):
    """Exploratory for PaySim synthetic financial datasets.

    Inherits from DatasetExploratory and adds PaySim-specific functional exploration
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
        # Override loader methods for PaySim-specific handling
        self._loader._parse_timestamps = self._skip_timestamp_parsing
        self._loader._normalize_target = self._skip_target_normalization

    def get_schema(self) -> PaySimSchema:
        """Return the PaySim schema.

        Returns:
            PaySimSchema with PaySim-specific column names and settings
        """
        return PaySimSchema()


    @staticmethod
    def _skip_timestamp_parsing(series: pd.Series) -> pd.Series:
        """Skip timestamp parsing for PaySim - step is already an integer."""
        return series

    @staticmethod
    def _skip_target_normalization(series: pd.Series) -> pd.Series:
        """Skip target normalization for PaySim - isFraud is already 0/1."""
        return series

    def run_functional(self) -> dict[str, object]:
        """Run structural and leakage-aware signal checks before committing to PaySim."""
        self._reporter.display_section("PaySim Synthetic Financial Datasets Exploration")

        loaded = self._loader.load()
        df = loaded.frame

        structure = self._inspect_structure(df)
        target = self._inspect_target(df)
        flagged_fraud = self._inspect_is_flagged_fraud(df)
        account_viability_sender = self._inspect_account_viability_sender(df)
        account_viability_namedest = self._inspect_account_viability_namedest(df)
        fraud_window_coverage = self._inspect_fraud_window_coverage(df)
        fraud_history_strata = self._inspect_fraud_history_strata(df)
        history_bucket_comparison = self._compare_history_bucket_distribution(df)

        split = self._simulate_chronological_split(df)
        checker = UnivariateSignalChecker()
        signal_with_leakage = checker.check_univariate_signal(df, target_column=TARGET_COLUMN, exclude_leakage=False)
        signal_without_leakage = checker.check_univariate_signal(df, target_column=TARGET_COLUMN, exclude_leakage=True)

        self._report_structure(structure)
        self._report_target(target)
        self._report_is_flagged_fraud(flagged_fraud)
        self._report_account_viability(account_viability_sender)
        self._report_account_viability(account_viability_namedest)
        self._report_split(split)
        self._report_signal(signal_with_leakage, signal_without_leakage)
        self._report_fraud_window_coverage(fraud_window_coverage)
        self._report_fraud_history_strata(fraud_history_strata)
        self._report_history_bucket_comparison(history_bucket_comparison)

        # Build summary
        summary = {
            "structure": structure,
            "target": target,
            "is_flagged_fraud": flagged_fraud,
            "account_viability_namedest": account_viability_namedest,
            "account_viability_sender": account_viability_sender,
            "fraud_window_coverage": fraud_window_coverage,
            "fraud_history_strata": fraud_history_strata.to_dict(),
            "history_bucket_comparison": history_bucket_comparison.to_dict(),
            "split": split,
            "signal_with_leakage": [r.__dict__ for r in signal_with_leakage],
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

    def _inspect_is_flagged_fraud(self, df: pd.DataFrame) -> dict[str, object]:
        """Check whether isFlaggedFraud overlaps with isFraud (potential artifact column)."""
        flagged = df["isFlaggedFraud"].astype(bool)
        fraud = df[TARGET_COLUMN].astype(bool)
        return {
            "n_flagged": int(flagged.sum()),
            "n_flagged_and_fraud": int((flagged & fraud).sum()),
            "n_flagged_not_fraud": int((flagged & ~fraud).sum()),
        }

    def _inspect_account_viability_sender(self, df: pd.DataFrame) -> dict[str, object]:
        """Check how many sender accounts support each candidate window length."""
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

    def _inspect_account_viability_namedest(self, df: pd.DataFrame) -> dict[str, object]:
        """Check how many accounts support each candidate window length, for both roles."""
        results: dict[str, object] = {}
        for role, column in (("sender", ACCOUNT_COLUMN), ("receiver", "nameDest")):
            counts = df.groupby(column).size()
            n_accounts = int(counts.shape[0])
            results[role] = {
                "n_accounts": n_accounts,
                "txn_per_account_median": float(counts.median()),
                "txn_per_account_p95": float(counts.quantile(0.95)),
                "pct_accounts_with_window": {
                    window: round(int((counts >= window).sum()) / n_accounts * 100, 2) for window in WINDOW_LENGTHS
                },
            }
        return results

    def _inspect_fraud_window_coverage(self, df: pd.DataFrame) -> dict[str, object]:
        """Check window viability and what fraction of fraud transactions actually have sufficient receiver history."""
        receiver_counts = df.groupby("nameDest").size()

        df = df.assign(receiver_history_len=df["nameDest"].map(receiver_counts))
        fraud_rows = df[df[TARGET_COLUMN] == 1]

        coverage = {
            window: round(float((fraud_rows["receiver_history_len"] >= window).mean()) * 100, 2)
            for window in WINDOW_LENGTHS
        }
        return {
            "n_relevant_rows": len(df),
            "n_relevant_receivers": int(receiver_counts.shape[0]),
            "receiver_history_median": float(receiver_counts.median()),
            "receiver_history_p95": float(receiver_counts.quantile(0.95)),
            "pct_fraud_with_window": coverage,
        }

    def _inspect_fraud_history_strata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Break down fraud transactions by receiver history-length bucket, checking
        whether thin-history fraud concentrates in particular amount or time ranges.
        """
        receiver_counts = df.groupby("nameDest").size()

        fraud_rows = df[df[TARGET_COLUMN] == 1].assign(history_len=lambda t: t["nameDest"].map(receiver_counts))
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
        Compare receiver history-length bucket distribution between fraud and
        non-fraud rows, to check whether thin history is fraud-specific or a
        general trait of the fraud-relevant transaction population.
        """
        receiver_counts = df.groupby("nameDest").size()
        df = df.assign(history_len=df["nameDest"].map(receiver_counts))
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
        Simulate the 70/15/15 receiver-level split; report fraud rate per split.
        Split by receiver (nameDest) so no receiver appears in multiple splits. The
        split never reads labels (receiver-size stratification only), so per-split
        fraud rates are an OUTCOME to inspect, not something the split forces equal.
        """
        from lstm_autoencoder.evaluation.config import SplitConfig
        from lstm_autoencoder.evaluation.receiver_split import ReceiverSplitter

        split_config = SplitConfig(train_fraction=ratios[0], validation_fraction=ratios[1])
        from lstm_autoencoder.data.paysim_synthetic_financial.causal_features import account_identity

        account = account_identity(df)
        groups = ReceiverSplitter(split_config, random_seed=42).split(account)

        parts = {name: df[account.isin(ids)] for name, ids in groups.items()}

        return {
            name: {
                "n": len(part),
                "n_receivers": int(account.loc[part.index].nunique()),
                "fraud_rate": float(part[TARGET_COLUMN].mean()),
            }
            for name, part in parts.items()
        }

    def _report_history_bucket_comparison(self, comparison: pd.DataFrame) -> None:
        self._reporter.display_section("History bucket: fraud vs normal distribution (lift)")
        self._reporter.display_table(comparison)

    def _report_fraud_history_strata(self, strata: pd.DataFrame) -> None:
        self._reporter.display_section("Fraud by receiver history-length bucket")
        self._reporter.display_table(strata)

    def _report_fraud_window_coverage(self, coverage: dict[str, object]) -> None:
        self._reporter.display_text(
            f"Rows (CASH_OUT/TRANSFER only): {coverage['n_relevant_rows']:,} | "
            f"unique receivers: {coverage['n_relevant_receivers']:,} | "
            f"receiver history median: {coverage['receiver_history_median']} | "
            f"p95: {coverage['receiver_history_p95']}"
        )
        self._reporter.display_section("Fraud transactions with receiver window >= W (%)")
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
        self._reporter.display_section("Fraud rate by transaction type")
        self._reporter.display_table(target["by_type"])

    def _report_is_flagged_fraud(self, flagged: dict[str, object]) -> None:
        if flagged["n_flagged"] == 0:
            self._reporter.display_text("isFlaggedFraud: no rows flagged, column has no signal.")
            return
        self._reporter.display_text(
            f"isFlaggedFraud: {flagged['n_flagged']} flagged, "
            f"{flagged['n_flagged_and_fraud']} overlap with isFraud, "
            f"{flagged['n_flagged_not_fraud']} do not."
        )

    def _report_account_viability(self, viability: dict[str, object]) -> None:
        if "sender" in viability and "receiver" in viability:
            for role in ("sender", "receiver"):
                role_data = viability[role]
                self._reporter.display_text(
                    f"{role.capitalize()} accounts: {role_data['n_accounts']:,} | "
                    f"txn/account median: {role_data['txn_per_account_median']} | "
                    f"p95: {role_data['txn_per_account_p95']}"
                )
                self._reporter.display_section(f"Accounts with window >= W (%) ({role})")
                self._reporter.display_table(
                    pd.DataFrame.from_dict(
                        role_data["pct_accounts_with_window"], orient="index", columns=["pct_accounts"]
                    )
                )
        else:
            self._reporter.display_text(
                f"Sender accounts: {viability['n_accounts']:,} | "
                f"txn/account median: {viability['txn_per_account_median']} | "
                f"p95: {viability['txn_per_account_p95']}"
            )
            self._reporter.display_section("Accounts with window >= W (%)")
            self._reporter.display_table(
                pd.DataFrame.from_dict(viability["pct_accounts_with_window"], orient="index", columns=["pct_accounts"])
            )

    def _report_split(self, split: dict[str, dict[str, float]]) -> None:
        self._reporter.display_section("Stratified split by receiver (fraud rate)")
        self._reporter.display_table(pd.DataFrame.from_dict(split, orient="index"))

    def _report_signal(
        self,
        with_leakage: list[SignalCheckResult],
        without_leakage: list[SignalCheckResult],
    ) -> None:
        self._reporter.display_section("Signal check: with vs without leakage columns")
        for label, results in (("With leakage", with_leakage), ("Without leakage", without_leakage)):
            self._reporter.display_section(label)
            self._reporter.display_table(
                pd.DataFrame.from_dict({r.feature: round(r.auc, 4) for r in results}, orient="index", columns=["auc"])
            )

        max_gap = max((abs(r.auc - 0.5) for r in without_leakage), default=0.0)
        if max_gap < SIGNAL_THRESHOLD:
            self._reporter.display_text(
                "[WARNING] No feature shows usable signal once leakage columns are "
                "excluded. Stop before building the full LSTM-AE pipeline — this is "
                "the same fatal pattern seen in the previous dataset."
            )
