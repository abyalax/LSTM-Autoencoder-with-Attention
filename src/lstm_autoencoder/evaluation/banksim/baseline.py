"""Ablation baselines for BankSim fraud detection: full vs no-history vs history-only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.banksim.causal_features import (
    ACCOUNT_CODE,
    HISTORY_BUCKET,
    HISTORY_LENGTH,
    IS_NEW_RECEIVER,
    LOG_AMOUNT_RATIO,
    ROLLING_LOG_RATIO_MAX_3,
    ROLLING_LOG_RATIO_MAX_5,
    ROLLING_LOG_RATIO_MEAN_3,
    ROLLING_LOG_RATIO_MEAN_5,
    STEP_DIFF,
    CausalFeatureBuilder,
)
from lstm_autoencoder.data.banksim.schema import BankSimSchema
from lstm_autoencoder.data.loader import DatasetLoader
from lstm_autoencoder.evaluation.config import ProbeConfig, SplitConfig
from lstm_autoencoder.evaluation.metrics import RankingMetrics
from lstm_autoencoder.evaluation.receiver_split import ReceiverSplitter
from lstm_autoencoder.model.config import SequenceConfig
from lstm_autoencoder.reporting.reporter import ReportWritter

HISTORY_STRATA_LABELS: tuple[str, ...] = ("1", "2-4", "5-9", ">=10")


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
    """Train logistic-regression and GBM ablations and report AUC/AP overall and per history bucket.

    Variants compared:
    - full: All features including history features
    - no_history: All features except history features (checks if history is just a proxy for easier signal)
    - history_only: Only history features (isolates the lift from history-based features)
    - full_gbm: GBM on all features (non-linear control)
    - full_gbm_no_step: GBM without the raw step column (measures how much advantage depends on hour-of-day)
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
        self._schema = BankSimSchema()
        self._loader = loader or DatasetLoader(dataset_config, self._schema)
        self._loader._parse_timestamps = self._skip_timestamp_parsing
        self._loader._normalize_target = self._skip_target_normalization
        self._reporter = reporter or ReportWritter(dataset_config)
        self._reporter.set_path_info_from_module()
        self._keep_account_code = keep_account_code

    @staticmethod
    def _skip_timestamp_parsing(series: pd.Series) -> pd.Series:
        """Skip timestamp parsing; BankSim's step is already an integer."""
        return series

    @staticmethod
    def _skip_target_normalization(series: pd.Series) -> pd.Series:
        """Skip target normalization; BankSim's fraud is already 0/1."""
        return series

    def fit_full_variant(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit the full-feature-set LR and return (test_idx, predicted_proba, labels)."""
        full_data = self._prepare_full_data()
        full_variant = self._define_variants(full_data)[0]  # Full variant
        test_idx = full_data.splits["test"]
        scores = self._fit_variant(full_variant, full_data)
        labels = full_data.labels[test_idx]
        return test_idx, scores[test_idx], labels

    def run(self) -> None:
        """Run all ablation variants and report results."""
        self._reporter.display_section("Ablation: LR full / no-history / history-only")

        full_data = self._prepare_full_data()
        variants = self._define_variants(full_data)

        results = {}
        for variant in variants:
            scores = self._fit_variant(variant, full_data)
            overall_metrics = self._compute_overall_metrics(scores, full_data)
            bucket_metrics = self._compute_bucket_metrics(scores, full_data)
            results[variant.name] = {
                "overall": overall_metrics,
                "by_bucket": bucket_metrics,
            }

        self._report_results(results)
        self._reporter.save_reports(
            text_filename="ablation_report.txt",
            json_filename="ablation_report.json",
        )

    def _prepare_full_data(self) -> FullFeatureData:
        """Load, build features, and prepare the full feature matrix."""
        loaded = self._loader.load()
        df = loaded.frame

        builder = CausalFeatureBuilder(self._dataset_config, self._schema)
        built = builder.build(df)

        # Split by account
        account = builder.account_identity(built)
        splitter = ReceiverSplitter(self._split_config, random_seed=42)
        groups = splitter.split(account)
        splits = {name: built[account.isin(ids)].index.values for name, ids in groups.items()}

        # Build feature matrix
        features = self._build_feature_matrix(built)
        labels = built[self._schema.target_column].to_numpy()
        buckets = built[HISTORY_BUCKET].to_numpy()
        account_code = built[ACCOUNT_CODE].to_numpy()

        return FullFeatureData(
            features=features,
            labels=labels,
            buckets=buckets,
            splits=splits,
            account_code=account_code,
        )

    def _build_feature_matrix(self, built: pd.DataFrame) -> pd.DataFrame:
        """Build the feature matrix excluding leakage columns."""
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
        if not self._keep_account_code:
            excluded.add(ACCOUNT_CODE)

        derived = CausalFeatureBuilder(self._dataset_config, self._schema).feature_columns(built)
        feature_dict: dict[str, pd.Series] = {}
        for column in built.columns:
            if column in excluded or column in derived:
                continue
            series = built[column]
            if pd.api.types.is_numeric_dtype(series):
                feature_dict[column] = series
            elif series.nunique() <= self._probe.max_categories:
                feature_dict[column] = pd.factorize(series)[0].astype("int32")

        for column in derived:
            if column == "history_bucket":
                feature_dict[column] = pd.factorize(built[column])[0].astype("int32")
            else:
                feature_dict[column] = built[column]

        features = pd.DataFrame(feature_dict, index=built.index)
        return features

    def _define_variants(self, full_data: FullFeatureData) -> list[AblationVariant]:
        """Define the ablation variants including GBM controls."""
        history_features = (HISTORY_LENGTH, IS_NEW_RECEIVER, HISTORY_BUCKET)
        rolling_features = (
            STEP_DIFF,
            LOG_AMOUNT_RATIO,
            ROLLING_LOG_RATIO_MEAN_3,
            ROLLING_LOG_RATIO_MAX_3,
            ROLLING_LOG_RATIO_MEAN_5,
            ROLLING_LOG_RATIO_MAX_5,
        )
        all_features = tuple(full_data.features.columns)

        no_history = tuple(col for col in all_features if col not in history_features)

        # Raw `step` lets a tree model recover hour-of-day, a simulator artefact (see `audit-pipeline`);
        # this control removes only that column to measure how much of the GBM's advantage depends on it.
        no_step = tuple(col for col in all_features if col != self._schema.timestamp_column)

        # rolling_only: test if sequential rolling features alone capture signal
        rolling_only = tuple(col for col in rolling_features if col in all_features)

        return [
            AblationVariant(name="full", columns=all_features),
            AblationVariant(name="no_history", columns=no_history),
            AblationVariant(name="history_only", columns=history_features),
            AblationVariant(name="rolling_only", columns=rolling_only),
            AblationVariant(name="full_gbm", columns=all_features),  # Non-linear control: HistGradientBoosting
            AblationVariant(name="full_gbm_no_step", columns=no_step),  # GBM without the raw `step` column
        ]

    def _fit_variant(self, variant: AblationVariant, full_data: FullFeatureData) -> np.ndarray:
        """Fit a variant on train/val and return scores on all rows."""
        train_idx = full_data.splits["train"]
        val_idx = full_data.splits["val"]

        X_train = full_data.features.loc[train_idx, list(variant.columns)]
        y_train = full_data.labels[train_idx]
        X_val = full_data.features.loc[val_idx, list(variant.columns)]
        y_val = full_data.labels[val_idx]

        # Use GBM for variants ending with _gbm, otherwise LogisticRegression
        if variant.name.endswith("_gbm"):
            model = HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.1,
                max_leaf_nodes=31,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=self._probe.random_seed,
            )
            model.fit(X_train, y_train)
            val_scores = model.predict_proba(X_val)[:, 1]

            # Retrain on train+val and score on all data
            train_val_idx = np.concatenate([train_idx, val_idx])
            X_train_val = full_data.features.loc[train_val_idx, list(variant.columns)]
            y_train_val = full_data.labels[train_val_idx]
            model.fit(X_train_val, y_train_val)
            all_scores = model.predict_proba(full_data.features[list(variant.columns)])[:, 1]
        else:
            # Use LogisticRegression for linear models
            model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self._probe.random_seed,
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model.fit(X_train_scaled, y_train)

            # Score on validation for threshold selection
            val_scores = model.predict_proba(X_val_scaled)[:, 1]

            # Retrain on train+val and score on all data
            train_val_idx = np.concatenate([train_idx, val_idx])
            X_train_val = full_data.features.loc[train_val_idx, list(variant.columns)]
            y_train_val = full_data.labels[train_val_idx]

            X_train_val_scaled = scaler.fit_transform(X_train_val)
            model.fit(X_train_val_scaled, y_train_val)

            X_all_scaled = scaler.transform(full_data.features[list(variant.columns)])
            all_scores = model.predict_proba(X_all_scaled)[:, 1]

            # Log coefficient importance for LR
            coef_df = pd.DataFrame({
                'feature': list(variant.columns),
                'coef_abs': np.abs(model.coef_[0]),
                'coef': model.coef_[0]
            }).sort_values('coef_abs', ascending=False)
            self._reporter.display_section(f"Feature coefficients ({variant.name})")
            self._reporter.display_table(coef_df.head(20))

        return all_scores

    def _compute_overall_metrics(self, scores: np.ndarray, full_data: FullFeatureData) -> dict[str, float]:
        """Compute overall AUC/AP on test set."""
        test_idx = full_data.splits["test"]
        y_true = full_data.labels[test_idx]
        y_score = scores[test_idx]

        return {
            "auc": RankingMetrics.roc_auc(y_score, y_true),
            "ap": RankingMetrics.average_precision(y_score, y_true),
            "prevalence": float(y_true.mean()),
        }

    def _compute_bucket_metrics(self, scores: np.ndarray, full_data: FullFeatureData) -> dict[str, dict[str, float]]:
        """Compute AUC/AP per history bucket on test set."""
        test_idx = full_data.splits["test"]
        test_buckets = full_data.buckets[test_idx]
        y_true = full_data.labels[test_idx]
        y_score = scores[test_idx]

        bucket_metrics = {}
        for label in HISTORY_STRATA_LABELS:
            mask = test_buckets == label
            if mask.sum() > 0:
                bucket_metrics[label] = {
                    "auc": RankingMetrics.roc_auc(y_score[mask], y_true[mask]),
                    "ap": RankingMetrics.average_precision(y_score[mask], y_true[mask]),
                    "n": int(mask.sum()),
                }
            else:
                bucket_metrics[label] = {"auc": 0.0, "ap": 0.0, "n": 0}

        return bucket_metrics

    def _report_results(self, results: dict[str, dict[str, dict]]) -> None:
        """Report the ablation results."""
        self._reporter.display_section("Overall metrics (test set)")
        overall_rows = []
        for variant_name, result in results.items():
            overall = result["overall"]
            overall_rows.append({
                "Variant": variant_name,
                "AUC": round(overall["auc"], 4),
                "AP": round(overall["ap"], 4),
                "Prevalence": round(overall["prevalence"], 4),
            })
        self._reporter.display_table(pd.DataFrame(overall_rows).set_index("Variant"))

        self._reporter.display_section("Per-bucket metrics (test set)")
        bucket_rows = []
        for variant_name, result in results.items():
            by_bucket = result["by_bucket"]
            for label, metrics in by_bucket.items():
                bucket_rows.append({
                    "Variant": variant_name,
                    "Bucket": label,
                    "AUC": round(metrics["auc"], 4),
                    "AP": round(metrics["ap"], 4),
                    "N": metrics["n"],
                })
        self._reporter.display_table(pd.DataFrame(bucket_rows))
