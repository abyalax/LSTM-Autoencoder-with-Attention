"""P3.5 hybrid classifier: LR baseline features plus LSTM-AE reconstruction error, fitted leak-free."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.evaluation.config import SplitConfig
from lstm_autoencoder.evaluation.metrics import RankingMetrics
from lstm_autoencoder.evaluation.paysim_synthetic_financial.baseline import (
    HISTORY_STRATA_LABELS,
    BaselineEvaluator,
    FullFeatureData,
)
from lstm_autoencoder.model.config import ModelConfig, SequenceConfig, TrainConfig
from lstm_autoencoder.model.pipeline import PaySimModelTrainingPipeline, SplitScoringData
from lstm_autoencoder.model.trainer import Trainer
from lstm_autoencoder.reporting.reporter import ReportWritter

BASELINE = "baseline_train"
LR_VAL_CONTROL = "lr_val_control"
HYBRID_VAL = "hybrid_val"
STACK_VAL = "stack_val"
HYBRID_TRAIN_NAIVE = "hybrid_train_naive"
NULL_HYBRID_VAL = "null_hybrid_val"
NULL_STACK_VAL = "null_stack_val"
AE_ONLY = "ae_error_only"
HYBRID_VAL_NORM = "hybrid_val_norm"  # Normalized error per history bucket

# (candidate, reference) pairs. Bootstrap CIs are computed for these.
BOOTSTRAP_COMPARISONS: tuple[tuple[str, str], ...] = (
    (HYBRID_VAL, LR_VAL_CONTROL),  # isolates the error feature: same fit data, same features minus error
    (HYBRID_VAL, BASELINE),  # vs. the production baseline (fit on train)
    (STACK_VAL, BASELINE),  # 2-parameter stack on top of the baseline score
)
# Point-estimate deltas only: how much the leaky protocol inflates, and the null-control noise floor.
DIAGNOSTIC_COMPARISONS: tuple[tuple[str, str], ...] = (
    (HYBRID_TRAIN_NAIVE, BASELINE),
    (NULL_HYBRID_VAL, LR_VAL_CONTROL),
    (NULL_STACK_VAL, BASELINE),
)


@dataclass(frozen=True)
class HybridClassifierConfig:
    """Settings for the hybrid classifier evaluation."""

    seeds: tuple[int, ...] = (42,)
    n_bootstrap: int = 200
    error_epsilon: float = 1e-8
    random_seed: int = 42
    fpr_targets: tuple[float, ...] = (0.01, 0.001)


@dataclass(frozen=True)
class _FittedLogistic:
    """A standardizer plus a fitted logistic regression."""

    scaler: StandardScaler
    model: LogisticRegression

    def score(self, x: np.ndarray) -> np.ndarray:
        """Return the logit: ranks like the probability but never saturates into ties."""
        return self.model.decision_function(self.scaler.transform(x))

    @property
    def last_coefficient(self) -> float:
        """Standardized coefficient of the LAST column (the reconstruction-error feature)."""
        return float(self.model.coef_[0][-1])


@dataclass(frozen=True)
class _SeedIndependentScores:
    """Baseline-side scores that do not depend on which AE seed is used."""

    baseline_val: np.ndarray
    baseline_test: np.ndarray
    control_test: np.ndarray


class HybridClassifierEvaluator:
    """Test whether LSTM-AE reconstruction error improves the `full` LR, without leakage.

    Why not just add the error as a column and fit on train: the AE trained on
    the normal train rows, so their error is in-sample (optimistically low)
    while val/test error is out-of-sample. A second stage fitted on the SAME
    train rows would learn from a feature distribution that test never shows.

    Leak-free protocol used here: the AE never saw `val` rows, so the second
    stage is fitted on `val` and judged on the untouched `test`.

    Variants (all scored on the same test rows):
      baseline_train      LR full, fit on train (the 0.8625 reference)
      lr_val_control      LR full, fit on val (same fit data as the hybrids)
      hybrid_val          LR full + log error, fit on val   [primary]
      stack_val           LR on [baseline logit, log error], fit on val [primary]
      hybrid_train_naive  LR full + log error, fit on train (LEAKY, diagnostic)
      null_*              same as hybrid_val / stack_val with the error column shuffled
      ae_error_only       reconstruction error alone

    Read hybrid_val against lr_val_control (identical fit data, only the error
    feature differs) and stack_val / hybrid_val against baseline_train.

    Caveat: `val` also picked the AE checkpoint (by validation reconstruction loss or,
    with `--selection-metric val_auc`, by labelled validation AUC -- the latter
    mildly favours the error feature on val). `test` is untouched, so the
    reported test numbers remain valid. Bootstrap resamples rows, not
    receivers, so its CIs are somewhat optimistic.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        sequence_config: SequenceConfig | None = None,
        model_config: ModelConfig | None = None,
        split_config: SplitConfig | None = None,
        config: HybridClassifierConfig | None = None,
        reporter: ReportWritter | None = None,
    ) -> None:
        """Bind the evaluator to its configuration."""
        self._dataset_config = dataset_config
        self._sequence_config = sequence_config or SequenceConfig()
        self._model_config = model_config or ModelConfig()
        self._split_config = split_config or SplitConfig()
        self._config = config or HybridClassifierConfig()
        self._reporter = reporter or ReportWritter(dataset_config)
        self._reporter.set_path_info_from_module()

    def run(self) -> dict[str, object]:
        """Score every split with each seed's checkpoint, fit all variants, report, and return the aggregate."""
        data = BaselineEvaluator(dataset_config=self._dataset_config).prepare_full_features()
        pipeline = PaySimModelTrainingPipeline(
            dataset_config=self._dataset_config,
            sequence_config=self._sequence_config,
            model_config=self._model_config,
            split_config=self._split_config,
        )
        scoring = pipeline.prepared_scoring_splits()
        self._assert_aligned(data, scoring)

        base = data.features.to_numpy(dtype="float64")
        x = {name: base[index] for name, index in data.splits.items()}
        y = {name: data.labels[index] for name, index in data.splits.items()}
        buckets = {name: data.buckets[index] for name, index in data.splits.items()}
        buckets_test = data.buckets[data.splits["test"]]

        baseline_fit = self._fit(x["train"], y["train"])
        control_fit = self._fit(x["val"], y["val"])
        shared = _SeedIndependentScores(
            baseline_val=baseline_fit.score(x["val"]),
            baseline_test=baseline_fit.score(x["test"]),
            control_test=control_fit.score(x["test"]),
        )

        self._reporter.display_text(
            f"train {len(y['train']):,} | val {len(y['val']):,} ({int(y['val'].sum()):,} fraud) | "
            f"test {len(y['test']):,} ({int(y['test'].sum()):,} fraud) | features {base.shape[1]}"
        )

        per_seed: list[dict[str, object]] = []
        for seed in self._config.seeds:
            errors = self._score_errors(pipeline, scoring, seed)
            result = self._run_seed(seed, x, y, buckets, buckets_test, errors, shared)
            per_seed.append(result)
            self._report_seed(result)

        self._record_run_config()
        aggregate = self._aggregate(per_seed)
        self._report_aggregate(aggregate)
        self._reporter.update_summary(
            {
                "hybrid_classifier": {
                    "window_length": self._sequence_config.window_length,
                    "architecture": self._architecture_tag(),
                    "seeds": list(self._config.seeds),
                    "n_bootstrap": self._config.n_bootstrap,
                    "per_seed": per_seed,
                    "aggregate": aggregate,
                }
            }
        )
        self._reporter.save_reports(
            text_filename="hybrid_classifier_report.txt",
            json_filename="hybrid_classifier_report.json",
        )
        return aggregate

    def _score_errors(
        self,
        pipeline: PaySimModelTrainingPipeline,
        scoring: dict[str, SplitScoringData],
        seed: int,
    ) -> dict[str, np.ndarray]:
        """Load this seed's checkpoint (no retraining) and score train/val/test windows."""
        checkpoint = pipeline.checkpoint_path(seed)
        if not checkpoint.exists():
            raise ValueError(
                f"Checkpoint not found: {checkpoint} ({self._checkpoint_name(seed)}). Train it first with "
                f"`train-model --window {self._sequence_config.window_length} --seeds {seed}` "
                "(add --bottleneck if that is the architecture you want)."
            )
        trainer = Trainer(self._model_config, TrainConfig(seed=seed), checkpoint.parent)
        model = trainer.build_eval_model(checkpoint)
        return {name: trainer.predict_errors(model, split.dataset) for name, split in scoring.items()}

    def _run_seed(
        self,
        seed: int,
        x: dict[str, np.ndarray],
        y: dict[str, np.ndarray],
        buckets: dict[str, np.ndarray],
        buckets_test: np.ndarray,
        errors: dict[str, np.ndarray],
        shared: _SeedIndependentScores,
    ) -> dict[str, object]:
        """Fit and score every variant for one AE seed."""
        # Compute train-normal statistics for leak-free normalization
        train_normal_mask = y["train"] == 0
        train_normal_log_errors = np.log(errors["train"][train_normal_mask] + self._config.error_epsilon)
        train_normal_stats: dict[str, tuple[float, float]] = {}
        for label in HISTORY_STRATA_LABELS:
            mask = (buckets["train"][train_normal_mask] == label)
            if mask.sum() > 0:
                bucket_log_errors = train_normal_log_errors[mask]
                train_normal_stats[label] = (float(bucket_log_errors.mean()), float(bucket_log_errors.std()))

        feature = {name: self._error_feature(values) for name, values in errors.items()}
        feature_norm = {name: self._error_feature(values, buckets[name], train_normal_stats) for name, values in errors.items()}
        rng = np.random.default_rng(self._config.random_seed + seed)
        shuffled = {name: rng.permutation(feature[name]) for name in ("val", "test")}

        hybrid_fit = self._fit(np.column_stack([x["val"], feature["val"]]), y["val"])
        hybrid_norm_fit = self._fit(np.column_stack([x["val"], feature_norm["val"]]), y["val"])
        stack_fit = self._fit(np.column_stack([shared.baseline_val, feature["val"]]), y["val"])
        naive_fit = self._fit(np.column_stack([x["train"], feature["train"]]), y["train"])
        null_hybrid_fit = self._fit(np.column_stack([x["val"], shuffled["val"]]), y["val"])
        null_stack_fit = self._fit(np.column_stack([shared.baseline_val, shuffled["val"]]), y["val"])

        test_scores = {
            BASELINE: shared.baseline_test,
            LR_VAL_CONTROL: shared.control_test,
            HYBRID_VAL: hybrid_fit.score(np.column_stack([x["test"], feature["test"]])),
            HYBRID_VAL_NORM: hybrid_norm_fit.score(np.column_stack([x["test"], feature_norm["test"]])),
            STACK_VAL: stack_fit.score(np.column_stack([shared.baseline_test, feature["test"]])),
            HYBRID_TRAIN_NAIVE: naive_fit.score(np.column_stack([x["test"], feature["test"]])),
            NULL_HYBRID_VAL: null_hybrid_fit.score(np.column_stack([x["test"], shuffled["test"]])),
            NULL_STACK_VAL: null_stack_fit.score(np.column_stack([shared.baseline_test, shuffled["test"]])),
            AE_ONLY: errors["test"],
        }
        labels = y["test"]
        variants = {name: self._metrics(scores, labels) for name, scores in test_scores.items()}
        deltas = {
            self._pair_name(candidate, reference): self._delta(variants[candidate], variants[reference])
            for candidate, reference in (*BOOTSTRAP_COMPARISONS, *DIAGNOSTIC_COMPARISONS)
        }
        bootstrap = self._bootstrap(test_scores, labels, seed) if self._config.n_bootstrap > 0 else {}
        return {
            "seed": seed,
            "variants": variants,
            "deltas": deltas,
            "bootstrap": bootstrap,
            "error_coefficients": {
                HYBRID_VAL: hybrid_fit.last_coefficient,
                HYBRID_VAL_NORM: hybrid_norm_fit.last_coefficient,
                STACK_VAL: stack_fit.last_coefficient,
                HYBRID_TRAIN_NAIVE: naive_fit.last_coefficient,
            },
            "by_bucket": self._by_bucket(test_scores, labels, buckets_test),
        }

    def _fit(self, x: np.ndarray, y: np.ndarray) -> _FittedLogistic:
        """Fit a standardized, class-balanced logistic regression (same recipe as the baseline)."""
        scaler = StandardScaler()
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self._config.random_seed)
        model.fit(scaler.fit_transform(x), y)
        return _FittedLogistic(scaler, model)

    def _error_feature(self, errors: np.ndarray, buckets: np.ndarray | None = None, train_normal_stats: dict[str, tuple[float, float]] | None = None) -> np.ndarray:
        """Log-transform the heavy-tailed reconstruction error (epsilon guards exact zeros).

        If buckets are provided, normalizes log-error per history bucket using train-normal statistics.
        This tests the hypothesis that AE error mimics non-linear history effects.

        Args:
            errors: Raw reconstruction errors
            buckets: History bucket labels (same length as errors)
            train_normal_stats: Pre-computed (mean, std) per bucket from train-normal rows.
                                 If None and buckets provided, computes from current split (legacy behavior).
        """
        error_array = np.asarray(errors, dtype="float64")
        # Log transform first (no clipping, allow negative log values)
        log_errors = np.log(error_array + self._config.error_epsilon)

        if buckets is not None:
            normalized = log_errors.copy()
            for label in HISTORY_STRATA_LABELS:
                mask = buckets == label
                if mask.sum() > 0:
                    if train_normal_stats is not None and label in train_normal_stats:
                        # Use train-normal statistics (leak-free protocol)
                        bucket_mean, bucket_std = train_normal_stats[label]
                    else:
                        # Legacy: compute from current split (may cause leakage if used on train/val)
                        bucket_log_errors = log_errors[mask]
                        bucket_mean = bucket_log_errors.mean()
                        bucket_std = bucket_log_errors.std()

                    if bucket_std > 1e-8:  # Avoid division by very small numbers
                        normalized[mask] = (log_errors[mask] - bucket_mean) / bucket_std
                    else:
                        # If std is too small, use log error minus mean
                        normalized[mask] = log_errors[mask] - bucket_mean
            log_errors = normalized

        return log_errors

    def _metrics(self, scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """AUC, AP and recall at the configured FPR targets."""
        out = {
            "auc": RankingMetrics.roc_auc(scores, labels),
            "ap": RankingMetrics.average_precision(scores, labels),
        }
        for fpr in self._config.fpr_targets:
            out[f"recall_at_fpr_{fpr:g}"] = RankingMetrics.recall_at_fpr(scores, labels, fpr)
        return out

    @staticmethod
    def _delta(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
        """Metric-by-metric difference candidate - reference."""
        return {key: candidate[key] - reference[key] for key in candidate}

    @staticmethod
    def _pair_name(candidate: str, reference: str) -> str:
        return f"{candidate}_vs_{reference}"

    def _bootstrap(self, scores: dict[str, np.ndarray], labels: np.ndarray, seed: int) -> dict[str, dict[str, float]]:
        """Paired, class-stratified bootstrap of delta AUC / delta AP / delta recall@FPR for the primary comparisons.

        Fraud and normal rows are resampled separately so every draw keeps the
        test prevalence; all variants are scored on the SAME draw (paired).
        """
        rng = np.random.default_rng(self._config.random_seed + 1_000 + seed)
        positives = np.flatnonzero(labels == 1)
        negatives = np.flatnonzero(labels == 0)
        names = sorted({name for pair in BOOTSTRAP_COMPARISONS for name in pair})
        # Initialize storage for all metrics
        metric_keys = ["auc", "ap"] + [f"recall_at_fpr_{fpr:g}" for fpr in self._config.fpr_targets]
        draws: dict[str, dict[str, list[float]]] = {
            self._pair_name(*pair): {key: [] for key in metric_keys} for pair in BOOTSTRAP_COMPARISONS
        }
        for _ in range(self._config.n_bootstrap):
            index = np.concatenate(
                [rng.choice(positives, size=len(positives)), rng.choice(negatives, size=len(negatives))]
            )
            resampled = labels[index]
            auc = {name: RankingMetrics.roc_auc(scores[name][index], resampled) for name in names}
            ap = {name: RankingMetrics.average_precision(scores[name][index], resampled) for name in names}
            recall_at_fpr = {
                fpr: {name: RankingMetrics.recall_at_fpr(scores[name][index], resampled, fpr) for name in names}
                for fpr in self._config.fpr_targets
            }
            for candidate, reference in BOOTSTRAP_COMPARISONS:
                pair_name = self._pair_name(candidate, reference)
                draws[pair_name]["auc"].append(auc[candidate] - auc[reference])
                draws[pair_name]["ap"].append(ap[candidate] - ap[reference])
                for fpr in self._config.fpr_targets:
                    key = f"recall_at_fpr_{fpr:g}"
                    draws[pair_name][key].append(recall_at_fpr[fpr][candidate] - recall_at_fpr[fpr][reference])

        summary: dict[str, dict[str, float]] = {}
        for pair_name, metrics in draws.items():
            auc_low, auc_high = np.percentile(metrics["auc"], [2.5, 97.5])
            ap_low, ap_high = np.percentile(metrics["ap"], [2.5, 97.5])
            summary[pair_name] = {
                "d_auc_ci_low": float(auc_low),
                "d_auc_ci_high": float(auc_high),
                "d_ap_ci_low": float(ap_low),
                "d_ap_ci_high": float(ap_high),
                "p_d_auc_le_0": float(np.mean(np.asarray(metrics["auc"]) <= 0.0)),
            }
            # Add recall@FPR CIs
            for fpr in self._config.fpr_targets:
                key = f"recall_at_fpr_{fpr:g}"
                recall_low, recall_high = np.percentile(metrics[key], [2.5, 97.5])
                summary[pair_name][f"d_{key}_ci_low"] = float(recall_low)
                summary[pair_name][f"d_{key}_ci_high"] = float(recall_high)
                summary[pair_name][f"p_d_{key}_le_0"] = float(np.mean(np.asarray(metrics[key]) <= 0.0))
        return summary

    def _by_bucket(
        self, scores: dict[str, np.ndarray], labels: np.ndarray, buckets: np.ndarray
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Test AUC/AP per receiver history bucket for the main variants and the AE error alone."""
        wanted = (BASELINE, LR_VAL_CONTROL, HYBRID_VAL, HYBRID_VAL_NORM, STACK_VAL, AE_ONLY)
        out: dict[str, dict[str, dict[str, float]]] = {}
        for name in wanted:
            rows: dict[str, dict[str, float]] = {}
            for label in HISTORY_STRATA_LABELS:
                mask = buckets == label
                if not mask.any():
                    continue
                rows[label] = {
                    "n": int(mask.sum()),
                    "n_fraud": int(labels[mask].sum()),
                    "auc": RankingMetrics.roc_auc(scores[name][mask], labels[mask]),
                    "ap": RankingMetrics.average_precision(scores[name][mask], labels[mask]),
                }
            out[name] = rows
        return out

    @staticmethod
    def _mean_std(values: list[float]) -> tuple[float, float]:
        """Mean and std (ddof=1 when more than one value)."""
        array = np.asarray(values, dtype="float64")
        return float(array.mean()), float(array.std(ddof=1 if len(array) > 1 else 0))

    def _record_run_config(self) -> None:
        """Write the resolved run configuration into the report (JSON + transcript).

        Unlike the free-text `command` field (what was typed), this records
        what the evaluator is configured to run: architecture, window, seeds
        and the checkpoint file each seed will use.
        """
        config = self._model_config
        run_config = {
            "architecture": config.architecture_tag,
            "window_length": self._sequence_config.window_length,
            "seeds": list(self._config.seeds),
            "model_config": {
                "attention": config.attention,
                "bottleneck": config.bottleneck,
                "attention_pooling": config.attention_pooling,
                "hidden_dim": config.hidden_dim,
                "bottleneck_dim": config.bottleneck_dim,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
            },
            "checkpoints": [self._checkpoint_name(seed) for seed in self._config.seeds],
        }
        self._reporter.update_summary({"run_config": run_config})
        self._reporter.display_text(
            f"Run config: architecture={config.architecture_tag} | "
            f"W={self._sequence_config.window_length} | seeds={list(self._config.seeds)}"
        )

    def _checkpoint_name(self, seed: int) -> str:
        """Generate the checkpoint filename for a given seed."""
        return f"lstm_ae_w{self._sequence_config.window_length}_{self._architecture_tag()}_seed{seed}_best.pt"

    def _aggregate(self, per_seed: list[dict[str, object]]) -> dict[str, object]:
        """Mean/std across seeds for every variant metric and every delta."""
        first_variants: dict[str, dict[str, float]] = per_seed[0]["variants"]  # type: ignore[assignment]
        variants: dict[str, dict[str, float]] = {}
        for name, metrics in first_variants.items():
            row: dict[str, float] = {}
            for metric in metrics:
                mean, std = self._mean_std([run["variants"][name][metric] for run in per_seed])  # type: ignore[index]
                row[f"{metric}_mean"], row[f"{metric}_std"] = mean, std
            variants[name] = row

        deltas: dict[str, dict[str, float]] = {}
        for candidate, reference in (*BOOTSTRAP_COMPARISONS, *DIAGNOSTIC_COMPARISONS):
            pair = self._pair_name(candidate, reference)
            row = {}
            # Aggregate point-estimate deltas (auc, ap, recall@FPR)
            first_deltas: dict[str, float] = per_seed[0]["deltas"][pair]  # type: ignore[assignment]
            for metric in first_deltas:
                mean, std = self._mean_std([run["deltas"][pair][metric] for run in per_seed])  # type: ignore[index]
                row[f"{metric}_mean"], row[f"{metric}_std"] = mean, std
            if (candidate, reference) in BOOTSTRAP_COMPARISONS and per_seed[0]["bootstrap"]:
                row["seeds_auc_ci_above_0"] = float(
                    sum(run["bootstrap"][pair]["d_auc_ci_low"] > 0.0 for run in per_seed)  # type: ignore[index]
                )
                # Add recall@FPR CI seed counts
                for fpr in self._config.fpr_targets:
                    key = f"recall_at_fpr_{fpr:g}"
                    row[f"seeds_{key}_ci_above_0"] = float(
                        sum(run["bootstrap"][pair][f"d_{key}_ci_low"] > 0.0 for run in per_seed)  # type: ignore[index]
                    )
                row["n_seeds"] = float(len(per_seed))
            deltas[pair] = row
        return {"n_seeds": len(per_seed), "variants": variants, "deltas": deltas}

    def _report_seed(self, result: dict[str, object]) -> None:
        """Print one seed's variant table, bootstrap CIs, coefficients and per-bucket AUC."""
        seed = result["seed"]
        self._reporter.display_section(f"Seed {seed}: test metrics per variant")
        self._show(pd.DataFrame(result["variants"]).T)
        self._reporter.display_section(f"Seed {seed}: delta vs reference (candidate - reference)")
        self._show(pd.DataFrame(result["deltas"]).T)
        if result["bootstrap"]:
            self._reporter.display_section(f"Seed {seed}: paired bootstrap 95% CI (n={self._config.n_bootstrap})")
            self._show(pd.DataFrame(result["bootstrap"]).T)
        self._reporter.display_text(
            "Standardized error-feature coefficient: "
            + " | ".join(f"{name} {value:+.4f}" for name, value in result["error_coefficients"].items())  # type: ignore[union-attr]
        )
        self._reporter.display_section(f"Seed {seed}: test AUC / AP by history bucket")
        rows = {
            f"{name} | bucket {bucket}": values
            for name, buckets in result["by_bucket"].items()  # type: ignore[union-attr]
            for bucket, values in buckets.items()
        }
        self._show(pd.DataFrame(rows).T)

    def _show(self, table: pd.DataFrame) -> None:
        """Display a table with numbers rounded so the grid stays readable."""
        self._reporter.display_table(table.round(5))

    def _report_aggregate(self, aggregate: dict[str, object]) -> None:
        """Print the across-seed summary and how to read it."""
        self._reporter.display_section("Aggregate across seeds: variants (mean / std)")
        self._show(pd.DataFrame(aggregate["variants"]).T)
        self._reporter.display_section("Aggregate across seeds: deltas (mean / std)")
        self._show(pd.DataFrame(aggregate["deltas"]).T)
        self._reporter.display_section(
            f"Hybrid Classifier (W={self._sequence_config.window_length}, "
            f"{self._architecture_tag()}, seeds={list(self._config.seeds)})"
        )
        self._reporter.display_text(
            "How to read: the error feature helps only if hybrid_val beats lr_val_control "
            "(same fit data) AND the paired CI of its delta AUC stays above 0 (seeds_auc_ci_above_0 == n_seeds), "
            "with the null_* deltas near 0. hybrid_train_naive minus baseline_train shows how much the "
            "in-sample (leaky) protocol would have inflated the gain."
        )

    @staticmethod
    def _assert_aligned(data: FullFeatureData, scoring: dict[str, SplitScoringData]) -> None:
        """Fail loudly if the AE pipeline and the LR frame disagree on rows, split by split."""
        for name, split in scoring.items():
            expected = data.splits[name]
            if not np.array_equal(split.row_index, expected):
                raise ValueError(
                    f"LSTM-AE and LR '{name}' row positions differ -- check both use the same "
                    "--dataset, --nrows, min_history_length=1 and the default split seed (42)."
                )
            if not np.array_equal(split.labels, data.labels[expected]):
                raise ValueError(f"Labels differ between LSTM-AE and LR '{name}' rows despite matching positions.")

    def _architecture_tag(self) -> str:
        """Short label for the current model variant."""
        return self._model_config.architecture_tag