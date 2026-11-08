"""Ablation baselines for PaySim fraud detection: full vs no-history vs history-only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.evaluation.config import ProbeConfig, SplitConfig
from lstm_autoencoder.data.loader import DatasetLoader
from lstm_autoencoder.data.paysim_synthetic_financial.causal_features import (
    ACCOUNT_CODE,
    HISTORY_BUCKET,
    HISTORY_LENGTH,
    IS_NEW_RECEIVER,
    CausalFeatureBuilder,
)
from lstm_autoencoder.data.paysim_synthetic_financial.schema import PaySimSchema
from lstm_autoencoder.evaluation.metrics import RankingMetrics
from lstm_autoencoder.reporting.reporter import ReportWritter

HISTORY_STRATA_LABELS: tuple[str, ...] = ("1", "2-4", "5-9", ">=10")
FRAUD_RATE_BINS: tuple[float, ...] = (0, 0.001, 0.01, 0.1, 1.0)
FRAUD_RATE_LABELS: tuple[str, ...] = ("very_low", "low", "medium", "high")


@dataclass(frozen=True)
class AblationVariant:
    """One feature-set ablation to train and compare."""

    name: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class FullFeatureData:
    """Leakage-free `full` LR feature matrix with aligned labels, buckets and receiver split."""

    features: pd.DataFrame
    labels: np.ndarray
    buckets: np.ndarray
    splits: dict[str, np.ndarray]
    # Factorized receiver id per row, aligned with `features`/`labels`/`buckets`.
    # Exposed so downstream evaluators (e.g. the hybrid classifier) can run a
    # receiver-clustered bootstrap instead of resampling individual rows.
    account_code: np.ndarray


class BaselineEvaluator:
    """Train logistic-regression ablations and report AUC/AP overall and per history bucket.

    Three variants are compared: the full feature set, the full set minus the
    causal history features (sanity check that history_length is not just a
    proxy for an easier signal), and the history features alone (isolates the
    lift found during exploration).
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        probe_config: ProbeConfig | None = None,
        split_config: SplitConfig | None = None,
        loader: DatasetLoader | None = None,
        reporter: ReportWritter | None = None,
        keep_account_code: bool = True,
    ) -> None:
        """Bind the evaluator to a dataset configuration and optional collaborators.

        Args:
            keep_account_code: If False, exclude account_code from features (it's a factorized ID, not behavioral).
        """
        self._dataset_config = dataset_config
        self._probe = probe_config or ProbeConfig()
        self._split_config = split_config or SplitConfig()
        self._schema = PaySimSchema()
        self._loader = loader or DatasetLoader(dataset_config, self._schema)
        self._loader._parse_timestamps = self._skip_timestamp_parsing
        self._loader._normalize_target = self._skip_target_normalization
        self._reporter = reporter or ReportWritter(dataset_config)
        self._reporter.set_path_info_from_module()
        self._keep_account_code = keep_account_code

    @staticmethod
    def _skip_timestamp_parsing(series: pd.Series) -> pd.Series:
        """Skip timestamp parsing; PaySim's step is already an integer."""
        return series

    @staticmethod
    def _skip_target_normalization(series: pd.Series) -> pd.Series:
        """Skip target normalization; PaySim's isFraud is already 0/1."""
        return series

    def fit_full_variant(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit the full-feature-set LR and return (test_idx, predicted_proba, labels).

        `test_idx` indexes rows of the causal-feature-built frame (same frame
        and same default random_seed=42 split as PaySimModelTrainingPipeline),
        so it lines up with that pipeline's `prepared_test_data()` test rows --
        for reuse by analyses that compare the two models row-for-row.
        """
        loaded = self._loader.load()
        built = CausalFeatureBuilder(self._dataset_config, self._schema).build(loaded.frame)
        labels = built[self._schema.target_column].to_numpy()
        splits = self._stratified_receiver_split(built)

        encoded = self._encode(built)
        modeling_columns = [c for c in encoded.columns if c != ACCOUNT_CODE]
        model, scaler = self._fit(encoded[modeling_columns], labels, splits["train"])

        test_idx = splits["test"]
        predicted = model.predict_proba(scaler.transform(encoded[modeling_columns].iloc[test_idx]))[:, 1]
        return test_idx, predicted, labels[test_idx]

    def prepare_full_features(self) -> FullFeatureData:
        """Return the `full` LR feature matrix with labels, buckets and the receiver split.

        Same frame, encoding and random_seed=42 split as `fit_full_variant`, so
        row positions line up with PaySimModelTrainingPipeline's batch rows.
        Exposed so hybrid analyses can refit LR on other splits or with extra columns.
        """
        loaded = self._loader.load()
        built = CausalFeatureBuilder(self._dataset_config, self._schema).build(loaded.frame)
        encoded = self._encode(built)
        modeling_columns = [c for c in encoded.columns if c != ACCOUNT_CODE]
        return FullFeatureData(
            features=encoded[modeling_columns],
            labels=built[self._schema.target_column].to_numpy(),
            buckets=built[HISTORY_BUCKET].to_numpy(),
            splits=self._stratified_receiver_split(built),
            account_code=built[ACCOUNT_CODE].to_numpy(),
        )

    def run(self) -> dict[str, object]:
        """Load data, build ablation variants, train and report each one."""
        loaded = self._loader.load()
        built = CausalFeatureBuilder(self._dataset_config, self._schema).build(loaded.frame)
        labels = built[self._schema.target_column].to_numpy()
        buckets = built[HISTORY_BUCKET].to_numpy()
        splits = self._stratified_receiver_split(built)

        self._reporter.display_section("ABLATION: full vs no-history vs history-only vs non-linear controls")
        self._reporter.display_text(
            f"Baris {len(built):,} | fraud {int(labels.sum()):,} ({labels.mean() * 100:.3f}%)"
        )

        encoded = self._encode(built)
        variants = self._build_variants(encoded)

        results: dict[str, object] = {}
        for variant in variants:
            columns = list(variant.columns)
            if variant.name == "full_gbm":
                model = self._fit_gbm(encoded[columns], labels, splits["train"])
                scores = self._score_gbm(model, encoded[columns], labels, splits)
                stratified = self._score_gbm_by_bucket(
                    model, encoded[columns], labels, splits["test"], buckets
                )
            else:
                model, scaler = self._fit(encoded[columns], labels, splits["train"])
                scores = self._score(model, scaler, encoded[columns], labels, splits)
                stratified = self._score_by_bucket(
                    model, scaler, encoded[columns], labels, splits["test"], buckets
                )
            results[variant.name] = {"overall": scores, "by_bucket": stratified}
            self._report_variant(variant.name, scores, stratified)

        self._reporter.update_summary({"ablation": results})
        self._reporter.save_reports(
            text_filename="ablation_report.txt",
            json_filename="ablation_report.json",
        )
        return results

    def _encode(self, built: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode transaction type and history_bucket, select numeric, leakage-free columns.

        Explicitly includes type_* dummy columns and history_bucket_* dummy columns.
        Account_code is included only if keep_account_code is True.
        """
        encoded = pd.get_dummies(
            built, columns=[self._schema.transaction_type_column, HISTORY_BUCKET], prefix=["type", "history_bucket"]
        )
        drop = (self._schema.target_column, "isFlaggedFraud", *self._schema.leakage_columns)
        # Get all numeric columns
        numeric = encoded.select_dtypes(include="number").columns
        # Build include list ensuring type_*, and history_bucket_* are present
        include_cols = set()
        for col in numeric:
            if col not in drop:
                include_cols.add(col)
        # Explicitly ensure account_code is included only if keep_account_code is True
        if self._keep_account_code and "account_code" in encoded.columns:
            include_cols.add("account_code")
        # Explicitly ensure type_* columns are included
        for col in encoded.columns:
            if col.startswith("type_"):
                include_cols.add(col)
        # Explicitly ensure history_bucket_* columns are included
        for col in encoded.columns:
            if col.startswith("history_bucket_"):
                include_cols.add(col)
        return encoded[sorted(include_cols)].fillna(0.0)

    def _build_variants(self, encoded: pd.DataFrame) -> list[AblationVariant]:
        """Define the full, no-history, history-only, and non-linear control feature sets."""
        history_columns = (HISTORY_LENGTH, IS_NEW_RECEIVER)
        # Exclude account_code from modeling columns if present
        modeling_columns = tuple(c for c in encoded.columns if c != ACCOUNT_CODE)
        # history_bucket_* is a one-hot of the same history signal (derived from
        # history_length); no_history must drop it too or the ablation leaks.
        history_bucket_columns = tuple(c for c in modeling_columns if c.startswith(f"{HISTORY_BUCKET}_"))
        no_history = tuple(
            c for c in modeling_columns if c not in history_columns and c not in history_bucket_columns
        )
        history_only = tuple(c for c in history_columns if c in modeling_columns)
        return [
            AblationVariant("full", modeling_columns),
            AblationVariant("no_history", no_history),
            AblationVariant("history_only", history_only),
            AblationVariant("full_gbm", modeling_columns),  # Non-linear control: HistGradientBoosting
        ]

    def _stratified_receiver_split(self, built: pd.DataFrame) -> dict[str, np.ndarray]:
        """Split receivers based on split_config fractions, stratified by per-receiver fraud rate."""
        receiver_column = self._schema.receiver_column
        target_column = self._schema.target_column
        stats = built.groupby(receiver_column).agg(
            n=(target_column, "size"), n_fraud=(target_column, "sum")
        )
        stats["fraud_rate"] = stats["n_fraud"] / stats["n"]
        bins = pd.cut(
            stats["fraud_rate"], bins=FRAUD_RATE_BINS, labels=FRAUD_RATE_LABELS, include_lowest=True
        ).fillna(FRAUD_RATE_LABELS[0])

        train_ids, rest_ids = train_test_split(
            stats.index, test_size=1 - self._split_config.train_fraction, random_state=self._probe.random_seed, stratify=bins
        )
        val_ratio = self._split_config.validation_fraction / (1 - self._split_config.train_fraction)
        val_ids, test_ids = train_test_split(
            rest_ids, test_size=1 - val_ratio, random_state=self._probe.random_seed, stratify=bins.loc[rest_ids]
        )

        receivers = built[receiver_column]
        return {
            "train": np.flatnonzero(receivers.isin(train_ids).to_numpy()),
            "val": np.flatnonzero(receivers.isin(val_ids).to_numpy()),
            "test": np.flatnonzero(receivers.isin(test_ids).to_numpy()),
        }

    def _fit(
        self, features: pd.DataFrame, labels: np.ndarray, train_idx: np.ndarray
    ) -> tuple[LogisticRegression, StandardScaler]:
        """Fit a standardized logistic regression on the training split."""
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features.iloc[train_idx])
        model = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=self._probe.random_seed
        )
        model.fit(x_train, labels[train_idx])
        return model, scaler

    def _score(
        self,
        model: LogisticRegression,
        scaler: StandardScaler,
        features: pd.DataFrame,
        labels: np.ndarray,
        splits: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Score the fitted model on the val and test splits."""
        scores: dict[str, float] = {}
        for name in ("val", "test"):
            idx = splits[name]
            predicted = model.predict_proba(scaler.transform(features.iloc[idx]))[:, 1]
            truth = labels[idx]
            scores[f"{name}_auc"] = RankingMetrics.roc_auc(predicted, truth)
            scores[f"{name}_ap"] = RankingMetrics.average_precision(predicted, truth)
        return scores

    def _score_by_bucket(
        self,
        model: LogisticRegression,
        scaler: StandardScaler,
        features: pd.DataFrame,
        labels: np.ndarray,
        test_idx: np.ndarray,
        buckets: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        """Score the test split, broken down by receiver history-length bucket."""
        predicted = model.predict_proba(scaler.transform(features.iloc[test_idx]))[:, 1]
        truth = labels[test_idx]
        bucket = buckets[test_idx]

        out: dict[str, dict[str, float]] = {}
        for label in HISTORY_STRATA_LABELS:
            mask = bucket == label
            if mask.sum() == 0:
                continue
            out[label] = {
                "n": int(mask.sum()),
                "auc": RankingMetrics.roc_auc(predicted[mask], truth[mask]),
                "ap": RankingMetrics.average_precision(predicted[mask], truth[mask]),
            }
        return out

    def _fit_gbm(
        self, features: pd.DataFrame, labels: np.ndarray, train_idx: np.ndarray
    ) -> HistGradientBoostingClassifier:
        """Fit a HistGradientBoostingClassifier on the training split (non-linear control).

        `early_stopping=True` carves `validation_fraction` off of `train_idx`
        internally (sklearn's own holdout), so this never touches our own
        `val`/`test` receivers -- same train-only fit contract as `_fit` (LR).
        A previous version accepted an unused `val_idx` argument and an
        inaccurate comment implying `val_idx` rows were used; both are removed
        here (P1 "Validasi baseline GBM"). Scoring (`_score_gbm`) already uses
        the same `RankingMetrics.roc_auc` / `average_precision` and the same
        `predict_proba[:, 1]` convention as the LR path (`_score`), so AUC/AP
        are computed identically for both models.
        """
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_leaf_nodes=31,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self._probe.random_seed,
        )
        model.fit(features.iloc[train_idx], labels[train_idx])
        return model

    def _score_gbm(
        self,
        model: HistGradientBoostingClassifier,
        features: pd.DataFrame,
        labels: np.ndarray,
        splits: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Score the fitted GBM model on the val and test splits."""
        scores: dict[str, float] = {}
        for name in ("val", "test"):
            idx = splits[name]
            predicted = model.predict_proba(features.iloc[idx])[:, 1]
            truth = labels[idx]
            scores[f"{name}_auc"] = RankingMetrics.roc_auc(predicted, truth)
            scores[f"{name}_ap"] = RankingMetrics.average_precision(predicted, truth)
        return scores

    def _score_gbm_by_bucket(
        self,
        model: HistGradientBoostingClassifier,
        features: pd.DataFrame,
        labels: np.ndarray,
        test_idx: np.ndarray,
        buckets: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        """Score the GBM test split, broken down by receiver history-length bucket."""
        predicted = model.predict_proba(features.iloc[test_idx])[:, 1]
        truth = labels[test_idx]
        bucket = buckets[test_idx]

        out: dict[str, dict[str, float]] = {}
        for label in HISTORY_STRATA_LABELS:
            mask = bucket == label
            if mask.sum() == 0:
                continue
            out[label] = {
                "n": int(mask.sum()),
                "auc": RankingMetrics.roc_auc(predicted[mask], truth[mask]),
                "ap": RankingMetrics.average_precision(predicted[mask], truth[mask]),
            }
        return out

    def _report_variant(
        self, name: str, scores: dict[str, float], stratified: dict[str, dict[str, float]]
    ) -> None:
        """Print one variant's overall and per-bucket scores."""
        self._reporter.display_section(f"Variant: {name}")
        self._reporter.display_table(pd.DataFrame({name: scores}).T)
        self._reporter.display_section(f"{name}: stratified by history bucket")
        self._reporter.display_table(pd.DataFrame(stratified).T)