"""Exploration of the financial fraud detection dataset before any modelling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.financial_fraud_detection.causal_features import CausalFeatureBuilder
from lstm_autoencoder.data.financial_fraud_detection.schema import FinancialFraudDetectionSchema
from lstm_autoencoder.data.financial_fraud_detection.sequence_profile import SequenceProfiler
from lstm_autoencoder.data.loader import DatasetLoader, LoadedDataset
from lstm_autoencoder.evaluation.financial_fraud_detection.metrices import RankingMetrics
from lstm_autoencoder.reporting.reporter import ReportWritter

FRAUD_RATE_TOLERANCE = 0.01
DRIFT_RATIO = 2.0
MIN_PERIOD_ROWS = 100
MIN_SPLIT_FRAUD = 100
WEAK_SIGNAL_DEVIATION = 0.03
LEAKAGE_DEVIATION = 0.40
LEAKAGE_MISSING_GAP = 0.5
MONTHLY_SPAN_DAYS = 90
MAX_LISTED_CATEGORIES = 25
TOP_CATEGORIES = 8


class FinancialFraudDetectionExplorer:
    """Run the dataset checks that decide whether the data suits the research."""

    def __init__(
        self,
        config: DatasetConfig,
        loader: DatasetLoader | None = None,
        reporter: ReportWritter | None = None,
    ) -> None:
        """Create an explorer; the loader and reporter default to standard ones."""
        self._config = config
        self._schema = FinancialFraudDetectionSchema()
        self._loader = loader or DatasetLoader(config, self._schema)
        self._reporter = reporter or ReportWritter(config)
        self._reporter.set_path_info_from_module()
        self._builder = CausalFeatureBuilder(config, self._schema)
        self._profiler = SequenceProfiler(config, self._schema, self._reporter.reporter)

    def run(self) -> dict[str, object]:
        """Execute every check, save the reports, and return the numeric summary."""
        self._reporter.summary.clear()
        loaded = self._loader.load()
        frame = loaded.frame
        self._report_structure(loaded)
        self._report_target(frame)
        self._report_amount(frame)

        built = self._builder.build(frame)
        self._reporter.update_summary(self._profiler.profile(built))
        aucs = self._report_signal(frame, built)
        self._report_leakage(frame, aucs)
        self._report_split(frame)
        self._report_summary()
        self._save_reports()
        return dict(self._reporter.summary)

    def _report_structure(self, loaded: LoadedDataset) -> None:
        frame = loaded.frame
        reporter = self._reporter.reporter
        reporter.display_section("1. STRUKTUR DATA")
        reporter.display_text(f"Sumber: {loaded.source}")
        reporter.display_text(
            f"Baris dibaca: {loaded.n_raw_rows:,} | dipakai: {len(frame):,} "
            f"| Kolom: {frame.shape[1]}"
        )
        if loaded.n_unreadable_timestamp:
            reporter.display_warning(
                f"{loaded.n_unreadable_timestamp} timestamp tidak terbaca (dibuang)."
            )
        if loaded.n_unreadable_target:
            reporter.display_warning(
                f"{loaded.n_unreadable_target} nilai target tidak terbaca (dibuang)."
            )

        missing = [c for c in self._schema.expected_columns if c not in frame.columns]
        extra = [c for c in frame.columns if c not in self._schema.expected_columns]
        if missing:
            reporter.display_warning(f"Kolom yang diharapkan tidak ada: {missing}")
        if extra:
            reporter.display_text(f"Kolom tambahan di luar catatan main.txt: {extra}")
            reporter.display_text(
                "  -> tinjau satu per satu: tersedia saat transaksi terjadi "
                "(real-time) atau turunan label?"
            )
        info = pd.DataFrame(
            {
                "dtype": frame.dtypes.astype(str),
                "n_missing": frame.isna().sum(),
                "pct_missing": (frame.isna().mean() * 100).round(3),
                "n_unique": frame.nunique(dropna=True),
            }
        )
        reporter.display_text()
        reporter.display_table(info)
        self._display_duplicates(frame)
        self._display_time_range(frame)
        self._reporter.update_summary(
            {
                "n_rows": len(frame),
                "columns": list(frame.columns),
                "extra_columns": extra,
            }
        )

    def _display_duplicates(self, frame: pd.DataFrame) -> None:
        if self._schema.id_column in frame.columns:
            duplicated_ids = int(frame[self._schema.id_column].duplicated().sum())
            self._reporter.display_text(
                f"\n{self._schema.id_column} duplikat: {duplicated_ids:,}"
            )
            if duplicated_ids:
                self._reporter.display_warning(f"{self._schema.id_column} tidak unik.")
        without_id = frame.drop(columns=[self._schema.id_column], errors="ignore")
        duplicated_rows = int(without_id.duplicated().sum())
        self._reporter.display_text(f"Baris duplikat (selain ID): {duplicated_rows:,}")

    def _display_time_range(self, frame: pd.DataFrame) -> None:
        stamps = frame[self._schema.timestamp_column]
        self._reporter.display_text(
            f"\nTimestamp: {stamps.min()}  ->  {stamps.max()}  "
            f"(rentang {stamps.max() - stamps.min()})"
        )
        ordered = bool(stamps.is_monotonic_increasing)
        self._reporter.display_text(f"Urut menurut waktu di file asli: {ordered}")

    def _report_target(self, frame: pd.DataFrame) -> None:
        reporter = self._reporter.reporter
        target = frame[self._schema.target_column]
        n_fraud = int(target.sum())
        rate = float(target.mean())
        reporter.display_section(f"2. TARGET ({self._schema.target_column})")
        reporter.display_text(
            f"Fraud: {n_fraud:,} dari {len(frame):,}  ->  {rate * 100:.3f}%   "
            f"(harapan ~{self._config.expected_fraud_rate * 100:.1f}%)"
        )
        reporter.display_text(f"AUPRC acak (tanpa kemampuan) = prevalensi = {rate:.4f}")
        if abs(rate - self._config.expected_fraud_rate) > FRAUD_RATE_TOLERANCE:
            reporter.display_warning("Rasio fraud menyimpang dari harapan.")
        reporter.display_text(
            "  Catatan: prevalensi fraud dunia nyata umumnya <1%. Pertimbangkan "
            "uji sensitivitas prevalensi pada Evaluation."
        )
        self._reporter.update_summary({"fraud_count": n_fraud, "fraud_rate": rate})

        for column in self._categorical_columns(frame):
            self._display_category_rates(frame, column, rate)
        self._display_temporal_rates(frame)

    def _categorical_columns(self, frame: pd.DataFrame) -> list[str]:
        excluded = {
            self._schema.target_column,
            self._schema.id_column,
            self._schema.timestamp_column,
            self._schema.sender_column,
            self._schema.receiver_column,
        }
        columns: list[str] = []
        for column in frame.columns:
            series = frame[column]
            if column in excluded or pd.api.types.is_numeric_dtype(series):
                continue
            if pd.api.types.is_datetime64_any_dtype(series):
                continue
            if series.nunique() <= 0.5 * len(frame):
                columns.append(column)
        return columns

    def _display_category_rates(
        self, frame: pd.DataFrame, column: str, prevalence: float
    ) -> None:
        target_name = self._schema.target_column
        grouped = frame.groupby(column, observed=True)[target_name]
        rates = grouped.agg(["size", "sum", "mean"])
        rates.columns = pd.Index(["n", "n_fraud", "fraud_rate"])
        rates["lift"] = (rates["fraud_rate"] / prevalence).round(2)
        rates = rates.sort_values("fraud_rate", ascending=False)

        support = max(50, int(0.0005 * len(frame)))
        supported = rates[rates["n"] >= support]
        shown = rates if len(rates) <= MAX_LISTED_CATEGORIES else supported
        self._reporter.display_text(f"\n-- {column}  ({len(rates)} nilai unik)")
        self._reporter.display_table(shown.head(MAX_LISTED_CATEGORIES).round(4))
        if bool((supported["fraud_rate"] >= 0.999).any()):
            self._reporter.display_warning(
                f"Kolom '{column}' punya kategori dengan 100% fraud "
                f"(dukungan>={support}), cek leakage."
            )

    def _display_temporal_rates(self, frame: pd.DataFrame) -> None:
        stamps = frame[self._schema.timestamp_column]
        span_days = (stamps.max() - stamps.min()).days
        monthly = span_days > MONTHLY_SPAN_DAYS
        periods = stamps.dt.to_period("M" if monthly else "W")
        grouped = frame.groupby(periods)[self._schema.target_column]
        rates = grouped.agg(["size", "sum", "mean"])
        rates.columns = pd.Index(["n", "n_fraud", "fraud_rate"])
        label = "bulan" if monthly else "minggu"
        self._reporter.display_text(f"\nFraud rate per periode ({label}):")
        self._reporter.display_table(rates.round(4))

        usable = rates[rates["n"] >= MIN_PERIOD_ROWS]["fraud_rate"]
        if len(usable) >= 2 and usable.min() > 0:  # noqa: SIM102
            if usable.max() / usable.min() > DRIFT_RATIO:
                self._reporter.display_warning(
                    f"Fraud rate antar periode berbeda >{DRIFT_RATIO:.0f}x, "
                    "ada indikasi drift temporal."
                )
        by_hour = frame.groupby(stamps.dt.hour)[self._schema.target_column].mean()
        self._reporter.display_text(
            f"\nFraud rate per jam: min {by_hour.min():.4f} "
            f"(jam {int(by_hour.idxmin())}), max {by_hour.max():.4f} "
            f"(jam {int(by_hour.idxmax())})"
        )
        self._reporter.update_summary({"fraud_rate_by_period": {
            str(period): float(rate) for period, rate in rates["fraud_rate"].items()
        }})

    def _report_amount(self, frame: pd.DataFrame) -> None:
        reporter = self._reporter.reporter
        amount = frame[self._schema.amount_column]
        target = frame[self._schema.target_column]
        reporter.display_section("3. AMOUNT")
        reporter.display_text(
            f"Negatif: {int((amount < 0).sum()):,} | Nol: {int((amount == 0).sum()):,} "
            f"| NaN: {int(amount.isna().sum()):,}"
        )
        described = amount.groupby(target).describe(percentiles=[0.5, 0.95, 0.99])
        described.index = described.index.map({0: "normal", 1: "fraud"})
        reporter.display_table(described.round(2))
        auc = RankingMetrics.roc_auc(amount.to_numpy(), target.to_numpy())
        reporter.display_text(f"AUC univariat amount (fraud vs normal): {auc:.4f}")
        if bool((amount < 0).any()):
            reporter.display_warning("Ada amount negatif.")

    def _report_signal(
        self, frame: pd.DataFrame, built: pd.DataFrame
    ) -> dict[str, float]:
        reporter = self._reporter.reporter
        reporter.display_section("5. SINYAL ANOMALI (AUC univariat, 0.5 = acak)")
        original = [
            column
            for column in frame.select_dtypes(include="number").columns
            if column not in (self._schema.target_column, self._schema.id_column)
        ]
        derived = self._builder.feature_columns(built)
        labels = built[self._schema.target_column].to_numpy()

        aucs: dict[str, float] = {}
        rows: list[dict[str, object]] = []
        for origin, columns in (("kolom asli", original), ("turunan kausal", derived)):
            for column in columns:
                values = built[column].to_numpy(dtype="float64")
                auc = RankingMetrics.roc_auc(values, labels)
                aucs[column] = auc
                rows.append(
                    {
                        "fitur": column,
                        "AUC": round(auc, 4),
                        "|AUC-0.5|": round(abs(auc - 0.5), 4),
                        "asal": origin,
                    }
                )
        table = pd.DataFrame(rows).sort_values("|AUC-0.5|", ascending=False)
        reporter.display_table(table.set_index("fitur"))
        if table["|AUC-0.5|"].max() < WEAK_SIGNAL_DEVIATION:
            reporter.display_warning(
                "Semua fitur numerik hampir tanpa sinyal univariat. Anomali mungkin "
                "hanya terlihat lewat kombinasi fitur atau sekuens."
            )
        self._reporter.update_summary({"univariate_auc": {
            name: (round(value, 4) if np.isfinite(value) else None)
            for name, value in aucs.items()
        }})
        return aucs

    def _report_leakage(self, frame: pd.DataFrame, aucs: dict[str, float]) -> None:
        reporter = self._reporter.reporter
        reporter.display_section("6. KANDIDAT LEAKAGE")
        flags: list[tuple[str, str]] = []
        target = frame[self._schema.target_column]
        for column in frame.columns:
            if column == self._schema.target_column:
                continue
            if column in self._schema.leakage_columns:
                flags.append((column, "ditetapkan sebagai leakage di konfigurasi"))
            elif "fraud" in column.lower():
                reason = "nama kolom memuat 'fraud', mungkin turunan label"
                flags.append((column, reason))
            auc = aucs.get(column, 0.5)
            if np.isfinite(auc) and abs(auc - 0.5) >= LEAKAGE_DEVIATION:
                flags.append((column, f"AUC univariat {auc:.3f} (sangat terpisah)"))
            if frame[column].isna().any():
                missing_by_class = frame[column].isna().groupby(target).mean()
                gap = abs(missing_by_class.get(1, 0.0) - missing_by_class.get(0, 0.0))
                if gap > LEAKAGE_MISSING_GAP:
                    reason = f"missingness berbeda tajam antar kelas ({gap:.2f})"
                    flags.append((column, reason))
        for column, reason in flags:
            reporter.display_warning(f"{column}: {reason}")
        if flags:
            reporter.display_text(
                "  Keputusan desain: kolom yang baru diketahui setelah transaksi "
                "atau turunan label JANGAN jadi input model."
            )
        else:
            reporter.display_text("Tidak ada kolom mencurigakan menurut heuristik.")
        self._reporter.update_summary({"leakage_flags": [
            {"column": column, "reason": reason} for column, reason in flags
        ]})

    def _report_split(self, frame: pd.DataFrame) -> None:
        reporter = self._reporter.reporter
        reporter.display_section("7. SIMULASI SPLIT KRONOLOGIS")
        ordered = frame.sort_values(self._schema.timestamp_column, kind="stable")
        n_rows = len(ordered)
        first = int(self._config.train_fraction * n_rows)
        second = int((self._config.train_fraction + self._config.validation_fraction) * n_rows)
        parts = {
            "train": ordered.iloc[:first],
            "val": ordered.iloc[first:second],
            "test": ordered.iloc[second:],
        }
        rows = [
            {
                "split": name,
                "n": len(part),
                "n_fraud": int(part[self._schema.target_column].sum()),
                "fraud_rate": round(float(part[self._schema.target_column].mean()), 4),
                "dari": str(part[self._schema.timestamp_column].min()),
                "sampai": str(part[self._schema.timestamp_column].max()),
            }
            for name, part in parts.items()
        ]
        table = pd.DataFrame(rows).set_index("split")
        reporter.display_table(table)
        seen = set(parts["train"][self._schema.sender_column].unique())
        cold = float((~parts["test"][self._schema.sender_column].isin(seen)).mean())
        reporter.display_text(
            f"\nTransaksi test dari akun yang tidak ada di train: {cold:.2%}"
        )
        if int(table["n_fraud"].min()) < MIN_SPLIT_FRAUD:
            reporter.display_warning(
                f"Ada split dengan <{MIN_SPLIT_FRAUD} fraud, metrik tidak stabil."
            )
        self._reporter.update_summary({"test_cold_start_share": round(cold, 4)})

    def _report_summary(self) -> None:
        self._reporter.display_section("RINGKASAN TEMUAN")
        warnings = self._reporter.get_warnings()
        if not warnings:
            self._reporter.display_text("Tidak ada temuan yang perlu perhatian.")
        for number, message in enumerate(warnings, start=1):
            self._reporter.display_text(f"{number}. {message}")
        self._reporter.update_summary({"warnings": list(warnings)})

    def _save_reports(self) -> None:
        self._reporter.save_reports(
            text_filename="dataset_report.txt",
            json_filename="dataset_report.json",
        )
