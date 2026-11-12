"""IEEE-CIS Fraud Detection-specific sequence profiler for assessing per-card history lengths."""

from __future__ import annotations

import pandas as pd

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.config import ExploratoryConfig
from lstm_autoencoder.data.ieee_fraud_detection.causal_features import (
    ACCOUNT_CODE,
    HISTORY_LENGTH,
    STEP_DIFF,
)
from lstm_autoencoder.data.ieee_fraud_detection.schema import IEEEFraudDetectionSchema
from lstm_autoencoder.reporting.console_report import ConsoleReporter

REFERENCE_WINDOW = 10
MIN_MEDIAN_HISTORY = 5
MIN_FULL_WINDOW_SHARE = 0.5
QUANTILES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.95, 1.0)


class SequenceProfiler:
    """Measure history lengths per card account and where fraud falls in them.

    IEEE-CIS-specific implementation that handles timestamp-based gaps.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        exploratory_config: ExploratoryConfig | None = None,
        schema: IEEEFraudDetectionSchema | None = None,
        reporter: ConsoleReporter | None = None,
    ) -> None:
        """Bind the profiler to a configuration, IEEE-CIS schema, and reporter."""
        self._dataset_config = dataset_config
        self._exploratory_config = exploratory_config or ExploratoryConfig()
        self._schema = schema or IEEEFraudDetectionSchema()
        self._reporter = reporter or ConsoleReporter()

    def profile(self, built: pd.DataFrame) -> dict[str, object]:
        """Report and return sequence statistics of a CausalFeatureBuilder frame."""
        target = built[self._schema.target_column]
        per_account = built.groupby(ACCOUNT_CODE, sort=False).size()
        fraud_rows = built[target == 1]

        self._reporter.display_section("4. VIABILITAS SEKUENS PER AKUN PENGIRIM")
        self._reporter.display_text(f"Akun pengirim unik: {len(per_account):,}")
        self._display_account_quantiles(per_account)
        full_window_share = self._display_window_table(built, fraud_rows, per_account)
        self._display_fraud_per_account(built, fraud_rows)
        self._display_gap_statistics(built)

        median_history = float(per_account.median())
        if median_history < MIN_MEDIAN_HISTORY:
            self._reporter.display_warning(
                f"Median transaksi per akun {median_history:.0f} "
                f"(<{MIN_MEDIAN_HISTORY}), sekuens per akun pendek sehingga "
                "LSTM sulit memberi nilai tambah."
            )
        if full_window_share < MIN_FULL_WINDOW_SHARE:
            self._reporter.display_warning(
                f"Kurang dari {MIN_FULL_WINDOW_SHARE:.0%} transaksi fraud punya "
                f"riwayat {REFERENCE_WINDOW} transaksi sebelumnya."
            )
        return {
            "accounts": len(per_account),
            "tx_per_account_median": median_history,
            f"fraud_with_full_window_w{REFERENCE_WINDOW}": full_window_share,
        }

    def _display_account_quantiles(self, per_account: pd.Series) -> None:
        values = per_account.quantile(list(QUANTILES)).astype(int)
        joined = " / ".join(f"{value:,}" for value in values)
        self._reporter.display_text(f"Transaksi per akun min/p25/median/p75/p95/max: {joined}")

    def _display_window_table(
        self,
        built: pd.DataFrame,
        fraud_rows: pd.DataFrame,
        per_account: pd.Series,
    ) -> float:
        rows: list[dict[str, float]] = []
        reference_share = 0.0
        for window in self._exploratory_config.window_sizes:
            has_window = built[HISTORY_LENGTH] >= window - 1
            fraud_share = float((fraud_rows[HISTORY_LENGTH] >= window - 1).mean())
            if window == REFERENCE_WINDOW:
                reference_share = fraud_share
            rows.append(
                {
                    "W": window,
                    "akun>=W (%)": round((per_account >= window).mean() * 100, 2),
                    "transaksi dg window penuh (%)": round(has_window.mean() * 100, 2),
                    "fraud dg window penuh (%)": round(fraud_share * 100, 2),
                }
            )
        self._reporter.display_text()
        self._reporter.display_table(pd.DataFrame(rows).set_index("W"))
        return reference_share

    def _display_fraud_per_account(self, built: pd.DataFrame, fraud_rows: pd.DataFrame) -> None:
        fraud_per_account = built.groupby(ACCOUNT_CODE, sort=False)[self._schema.target_column].sum()
        with_fraud = int((fraud_per_account > 0).sum())
        share = (fraud_per_account > 0).mean() * 100
        first_share = (fraud_rows[HISTORY_LENGTH] == 0).mean() * 100
        self._reporter.display_text()
        repeated = int((fraud_per_account > 1).sum())
        reporter = self._reporter
        reporter.display_text(f"Akun dengan >=1 fraud: {with_fraud:,} ({share:.2f}%)")
        reporter.display_text(f"Akun dengan >=2 fraud: {repeated:,}")
        reporter.display_text(f"Fraud pada transaksi PERTAMA akun: {first_share:.2f}%")

    def _display_gap_statistics(self, built: pd.DataFrame) -> None:
        gaps = built[STEP_DIFF]
        ties = int((gaps == 0).sum())
        self._reporter.display_text(f"Transaksi dengan selisih timestamp 0 dari sebelumnya: {ties:,}")
        by_class = gaps.groupby(built.loc[gaps.index, self._schema.target_column])
        described = by_class.describe(percentiles=[0.5, 0.95]).round(2)
        described.index = described.index.map({0: "normal", 1: "fraud"})
        self._reporter.display_text("\nStatistik selisih timestamp per kelas:")
        self._reporter.display_table(described)
