"""Check whether the fraud label can be learned from PaySim features."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.evaluation.config import ProbeConfig, SplitConfig
from lstm_autoencoder.data.paysim_synthetic_financial.schema import PaySimSchema
from lstm_autoencoder.evaluation.paysim_synthetic_financial.metrics import (
    UnivariateSignalChecker,
)
from lstm_autoencoder.evaluation.signal_probe import BaseSignalProbe, SignalVerdict
from lstm_autoencoder.reporting.reporter import ReportWritter

if TYPE_CHECKING:
    from lstm_autoencoder.data.loader import DatasetLoader, LoadedDataset


class PaySimSignalProbe(BaseSignalProbe):
    """Train a strong supervised model on PaySim data and check label learnability."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        probe_config: ProbeConfig | None = None,
        split_config: SplitConfig | None = None,
        loader: "DatasetLoader | None" = None,  # noqa: UP037
        reporter: ReportWritter | None = None,
    ) -> None:
        """Create a probe; optional collaborators default to standard ones."""
        super().__init__(dataset_config, probe_config, split_config, loader, reporter)
        self._schema = PaySimSchema()
        if self._loader is None:
            from lstm_autoencoder.data.loader import DatasetLoader

            self._loader = DatasetLoader(dataset_config, self._schema)
        # Override loader methods for PaySim-specific handling
        self._loader._parse_timestamps = self._skip_timestamp_parsing
        self._loader._normalize_target = self._skip_target_normalization

    @staticmethod
    def _skip_timestamp_parsing(series: pd.Series) -> pd.Series:
        """Skip timestamp parsing for PaySim - step is already an integer."""
        return series

    @staticmethod
    def _skip_target_normalization(series: pd.Series) -> pd.Series:
        """Skip target normalization for PaySim - isFraud is already 0/1."""
        return series

    def _get_schema(self) -> PaySimSchema:
        """Return the schema for the dataset."""
        return self._schema

    def _prepare_data(self, loaded: LoadedDataset) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
        """Prepare the data for signal probe using stratified split by receiver."""
        from sklearn.model_selection import train_test_split

        df = loaded.frame

        # Get unique receivers and their fraud rates
        receiver_stats = (
            df.groupby("nameDest")
            .agg(
                n_transactions=(self._schema.target_column, "size"),
                n_fraud=(self._schema.target_column, "sum"),
            )
            .reset_index()
        )
        receiver_stats["fraud_rate"] = receiver_stats["n_fraud"] / receiver_stats["n_transactions"]

        # Split receivers into train/val/test (stratified by fraud rate)
        ratios = (0.7, 0.15, 0.15)
        stratify_bins = pd.cut(
            receiver_stats["fraud_rate"],
            bins=[0, 0.001, 0.01, 0.1, 1.0],
            labels=["very_low", "low", "medium", "high"],
            include_lowest=True,
        ).fillna("very_low")
        receivers_train, receivers_valtest = train_test_split(
            receiver_stats["nameDest"],
            test_size=ratios[1] + ratios[2],
            random_state=42,
            stratify=stratify_bins,
        )

        val_ratio = ratios[1] / (ratios[1] + ratios[2])
        receivers_valtest_df = receiver_stats[receiver_stats["nameDest"].isin(receivers_valtest)]
        stratify_bins_valtest = pd.cut(
            receivers_valtest_df["fraud_rate"],
            bins=[0, 0.001, 0.01, 0.1, 1.0],
            labels=["very_low", "low", "medium", "high"],
            include_lowest=True,
        ).fillna("very_low")
        receivers_val, receivers_test = train_test_split(
            receivers_valtest_df["nameDest"],
            test_size=1 - val_ratio,
            random_state=42,
            stratify=stratify_bins_valtest,
        )

        # Assign transactions to splits based on receiver
        train_mask = df["nameDest"].isin(receivers_train)
        val_mask = df["nameDest"].isin(receivers_val)
        test_mask = df["nameDest"].isin(receivers_test)

        # Create split indices
        train_indices = np.where(train_mask.values)[0]
        val_indices = np.where(val_mask.values)[0]
        test_indices = np.where(test_mask.values)[0]

        splits = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }

        # Combine all splits with original order for consistent indexing
        ordered = df.reset_index(drop=True)
        labels = ordered[self._schema.target_column].to_numpy()

        return ordered, labels, splits

    def _display_additional_info(self, ordered: pd.DataFrame, labels: np.ndarray) -> None:
        """Display additional PaySim-specific information (placeholder for future extensions)."""
        pass

    def _build_features(self, ordered: pd.DataFrame) -> tuple[pd.DataFrame, list[bool]]:
        """Build features from the ordered dataframe.

        For PaySim, this includes one-hot encoding transaction type.
        """
        # Add one-hot encoded transaction type
        df_encoded = pd.get_dummies(ordered, columns=["type"], prefix="type")

        # Prepare features: use numeric columns excluding leakage and target
        leakage_cols = self._schema.leakage_columns
        feature_cols = [
            col
            for col in df_encoded.columns
            if col not in (self._schema.target_column, "isFlaggedFraud") and col not in leakage_cols
        ]

        features = df_encoded[feature_cols]
        categorical = [col.startswith("type_") for col in feature_cols]

        return features, categorical

    def run(self) -> SignalVerdict:
        """Run the probe with PaySim-specific univariate signal check."""
        loaded = self._loader.load()
        ordered, labels, splits = self._prepare_data(loaded)

        self._reporter.display_section("PROBE SINYAL: apakah isFraud bisa dipelajari?")
        self._reporter.display_text(
            f"Baris {len(ordered):,} | fraud {int(labels.sum()):,} ({labels.mean() * 100:.3f}%)"
        )

        # Display split statistics
        train_mask = np.zeros(len(ordered), dtype=bool)
        val_mask = np.zeros(len(ordered), dtype=bool)
        test_mask = np.zeros(len(ordered), dtype=bool)
        train_mask[splits["train"]] = True
        val_mask[splits["val"]] = True
        test_mask[splits["test"]] = True

        self._reporter.display_text(
            f"Train: {len(splits['train']):,} rows, fraud rate {labels[train_mask].mean() * 100:.3f}%"
        )
        self._reporter.display_text(
            f"Val: {len(splits['val']):,} rows, fraud rate {labels[val_mask].mean() * 100:.3f}%"
        )
        self._reporter.display_text(
            f"Test: {len(splits['test']):,} rows, fraud rate {labels[test_mask].mean() * 100:.3f}%"
        )

        # Check univariate signal first
        signal_results = UnivariateSignalChecker().check_univariate_signal(
            ordered, target_column=self._schema.target_column, exclude_leakage=True
        )
        self._reporter.display_text("\nUnivariate signal check (without leakage):")
        for result in signal_results[:5]:  # Show top 5
            self._reporter.display_text(f"  {result.feature}: AUC = {result.auc:.4f}")

        has_signal = UnivariateSignalChecker().has_usable_signal(signal_results)

        if not has_signal:
            self._reporter.display_text(
                "\n[WARNING] Tidak ada sinyal univariat yang usable. Model supervised tidak akan berhasil."
            )
            verdict = SignalVerdict.NO_SIGNAL
            self._reporter.save_reports(
                text_filename="signal_report.txt",
                json_filename="signal_report.json",
            )
            return verdict

        # Display additional info and build features
        self._display_additional_info(ordered, labels)
        feature_df, categorical = self._build_features(ordered)
        names = list(feature_df.columns)
        self._reporter.display_text(f"\nFitur ({len(names)}): {names}")

        # Evaluate using base class method with custom splits
        results = self._evaluate_with_splits(feature_df, categorical, labels, splits)
        verdict = self._decide(results)
        self._reporter.save_reports(
            text_filename="signal_report.txt",
            json_filename="signal_report.json",
        )
        return verdict

    def _decide(self, results: dict[str, dict[str, float]]) -> SignalVerdict:
        """Decide the verdict based on evaluation results."""
        real = results["label asli"]
        control_auc = results["label diacak"]["test_auc"]
        auc, ap = real["test_auc"], real["test_ap"]
        prevalence = real["test_prevalence"]
        reporter = self._reporter.reporter
        reporter.display_text("\nKESIMPULAN")
        summary = (
            f"AUC test {auc:.3f}, AP {ap:.4f} (prevalensi {prevalence:.4f}), kontrol label-acak AUC {control_auc:.3f}."
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
            reporter.display_text("  Sinyal sangat lemah, hasil model deep kemungkinan tidak bermakna.")
            return SignalVerdict.WEAK_SIGNAL
        reporter.display_text(f"  {summary}")
        reporter.display_text("  Ada sinyal yang dapat dipelajari, lanjut ke desain model.")
        return SignalVerdict.LEARNABLE

    def _evaluate_with_splits(
        self,
        features: pd.DataFrame,
        categorical: list[bool],
        labels: np.ndarray,
        splits: dict[str, np.ndarray],
    ) -> dict[str, dict[str, float]]:
        """Evaluate the features using supervised model with pre-computed splits."""
        from sklearn.ensemble import HistGradientBoostingClassifier

        rng = np.random.default_rng(self._probe.random_seed)
        train = splits["train"]
        if len(train) > self._probe.max_train_rows:
            chosen = rng.choice(train, self._probe.max_train_rows, replace=False)
            train = np.sort(chosen)

        self._reporter.display_text(
            f"\nBatas atas supervised | train {len(train):,} val {len(splits['val']):,} test {len(splits['test']):,}"
        )
        shuffled = rng.permutation(labels[train])
        results = {
            "label asli": self._fit_and_score(features, categorical, train, labels[train], splits, labels),
            "label diacak": self._fit_and_score(features, categorical, train, shuffled, splits, labels),
        }
        rows = {name: {key: round(value, 4) for key, value in scores.items()} for name, scores in results.items()}
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
        from sklearn.ensemble import HistGradientBoostingClassifier
        from lstm_autoencoder.evaluation.metrics import RankingMetrics

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
            if name == "train":
                continue
            predicted = model.predict_proba(features.iloc[index])[:, 1]
            truth = labels[index]
            scores[f"{name}_auc"] = RankingMetrics.roc_auc(predicted, truth)
            scores[f"{name}_ap"] = RankingMetrics.average_precision(predicted, truth)
            scores[f"{name}_prevalence"] = float(truth.mean())
        return scores
