"""Does LSTM-AE reconstruction error add anything on top of a strong tabular baseline? (torch-free)

The reference is a gradient-boosting model, not logistic regression: on BankSim the GBM
already reaches AUC ~0.998, so beating a weak LR would say nothing about the AE.

Leak-free protocol (same idea as the PaySim hybrid evaluation): the AE trained on normal
TRAIN windows, so its error on train is in-sample and unusable. The second stage is
therefore fit on VAL and judged on the untouched TEST split.

Variants, all scored on the same test rows:
  lr_train / gbm_train   tabular baselines fit on train (the production references)
  gbm_val_control        the same GBM fit on VAL with the SAME features, no AE error
  hybrid_gbm_val         that GBM plus the AE error as one more feature   [primary]
  stack_ae               LR on [gbm_train score, AE error], fit on val
  null_*                 hybrid / stack with the error column shuffled (noise floor)
  ae_only                the reconstruction error alone

Primary reading: hybrid_gbm_val vs gbm_val_control (identical fit data and features, only
the error differs) and stack_ae vs gbm_train. Delta CIs resample whole CUSTOMERS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from lstm_autoencoder.evaluation.bootstrap import WeightedRankingBootstrap
from lstm_autoencoder.evaluation.metrics import RankingMetrics

LR_TRAIN = "lr_train"
GBM_TRAIN = "gbm_train"
GBM_VAL_CONTROL = "gbm_val_control"
HYBRID_GBM_VAL = "hybrid_gbm_val"
STACK_AE = "stack_ae"
NULL_HYBRID = "null_hybrid_gbm_val"
NULL_STACK = "null_stack"
AE_ONLY = "ae_only"

COMPARISONS: tuple[tuple[str, str], ...] = (
    (HYBRID_GBM_VAL, GBM_VAL_CONTROL),  # primary: isolates the error feature
    (STACK_AE, GBM_TRAIN),  # 2-parameter stack on top of the production baseline
    (NULL_HYBRID, GBM_VAL_CONTROL),  # noise floors
    (NULL_STACK, GBM_TRAIN),
)


@dataclass(frozen=True)
class SplitArrays:
    """Feature matrix, labels and customer ids of one split, row-aligned with the AE windows."""

    x: np.ndarray
    y: np.ndarray
    customers: np.ndarray


@dataclass(frozen=True)
class HybridInputs:
    """The three splits of one evaluation."""

    train: SplitArrays
    val: SplitArrays
    test: SplitArrays


@dataclass(frozen=True)
class HybridScoringConfig:
    """Knobs of the hybrid evaluation."""

    n_bootstrap: int = 1000
    random_seed: int = 42
    error_epsilon: float = 1e-8
    fpr_targets: tuple[float, ...] = (0.01, 0.001)


class HybridScorer:
    """Fit the baselines once, then score any number of AE seeds against them."""

    def __init__(self, inputs: HybridInputs, config: HybridScoringConfig | None = None) -> None:
        """Fit the seed-independent baselines on train (and the no-error control on val)."""
        self._inputs = inputs
        self._config = config or HybridScoringConfig()
        train, val, test = inputs.train, inputs.val, inputs.test
        gbm_train = self._fit_gbm(train.x, train.y)
        lr_train = self._fit_lr(train.x, train.y)
        gbm_control = self._fit_gbm(val.x, val.y)
        self._gbm_train_val = gbm_train.decision_function(val.x)
        self._shared_test = {
            GBM_TRAIN: gbm_train.decision_function(test.x),
            LR_TRAIN: lr_train.score(test.x),
            GBM_VAL_CONTROL: gbm_control.decision_function(test.x),
        }
        self._gbm_train_test = self._shared_test[GBM_TRAIN]

    def score_seed(self, seed: int, errors: dict[str, np.ndarray]) -> dict[str, object]:
        """Score one AE seed. `errors` maps "val" / "test" to that seed's anomaly score per row."""
        config = self._config
        val, test = self._inputs.val, self._inputs.test
        feature = {name: self._log_error(errors[name]) for name in ("val", "test")}
        rng = np.random.default_rng(config.random_seed + seed)
        shuffled = {name: rng.permutation(feature[name]) for name in ("val", "test")}

        hybrid = self._fit_gbm(np.column_stack([val.x, feature["val"]]), val.y)
        null_hybrid = self._fit_gbm(np.column_stack([val.x, shuffled["val"]]), val.y)
        stack = self._fit_lr(np.column_stack([self._gbm_train_val, feature["val"]]), val.y)
        null_stack = self._fit_lr(np.column_stack([self._gbm_train_val, shuffled["val"]]), val.y)

        scores = {
            **self._shared_test,
            HYBRID_GBM_VAL: hybrid.decision_function(np.column_stack([test.x, feature["test"]])),
            NULL_HYBRID: null_hybrid.decision_function(np.column_stack([test.x, shuffled["test"]])),
            STACK_AE: stack.score(np.column_stack([self._gbm_train_test, feature["test"]])),
            NULL_STACK: null_stack.score(np.column_stack([self._gbm_train_test, shuffled["test"]])),
            AE_ONLY: errors["test"],
        }
        variants = {name: self._metrics(values, test.y) for name, values in scores.items()}
        deltas = {
            f"{candidate}_vs_{reference}": {key: variants[candidate][key] - variants[reference][key] for key in variants[candidate]}
            for candidate, reference in COMPARISONS
        }
        bootstrap = self._bootstrap(scores, seed) if config.n_bootstrap > 0 else {}
        return {"seed": seed, "variants": variants, "deltas": deltas, "bootstrap": bootstrap}

    def _log_error(self, errors: np.ndarray) -> np.ndarray:
        """Log-transform the heavy-tailed reconstruction error (epsilon guards exact zeros)."""
        return np.log(np.asarray(errors, dtype="float64") + self._config.error_epsilon)

    def _metrics(self, scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        out = {
            "auc": float(RankingMetrics.roc_auc(scores, labels)),
            "ap": float(RankingMetrics.average_precision(scores, labels)),
        }
        for fpr in self._config.fpr_targets:
            out[f"recall_at_fpr_{fpr:g}"] = float(RankingMetrics.recall_at_fpr(scores, labels, fpr))
        return out

    def _bootstrap(self, scores: dict[str, np.ndarray], seed: int) -> dict[str, dict[str, float]]:
        """Paired customer-cluster bootstrap of delta AUC / delta AP for every comparison."""
        test = self._inputs.test
        names = sorted({name for pair in COMPARISONS for name in pair})
        engine = WeightedRankingBootstrap(test.y, {name: scores[name] for name in names}, fpr_targets=())
        _, inverse = np.unique(test.customers, return_inverse=True)
        n_customers = int(inverse.max()) + 1
        rng = np.random.default_rng(self._config.random_seed + 1_000 + seed)
        draws: dict[str, dict[str, list[float]]] = {f"{c}_vs_{r}": {"auc": [], "ap": []} for c, r in COMPARISONS}
        for _ in range(self._config.n_bootstrap):
            metrics = engine.metrics(WeightedRankingBootstrap.cluster_weights(inverse, n_customers, rng))
            for candidate, reference in COMPARISONS:
                for key in ("auc", "ap"):
                    draws[f"{candidate}_vs_{reference}"][key].append(metrics[candidate][key] - metrics[reference][key])
        summary: dict[str, dict[str, float]] = {}
        for pair, values in draws.items():
            entry: dict[str, float] = {}
            for key in ("auc", "ap"):
                array = np.asarray(values[key], dtype="float64")
                low, high = np.nanpercentile(array, [2.5, 97.5])
                entry[f"d_{key}_ci_low"] = float(low)
                entry[f"d_{key}_ci_high"] = float(high)
                entry[f"p_d_{key}_le_0"] = float(np.nanmean(array <= 0.0))
            summary[pair] = entry
        return summary

    def _fit_gbm(self, x: np.ndarray, y: np.ndarray) -> HistGradientBoostingClassifier:
        """Same hyperparameters as the `full_gbm` ablation baseline."""
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_leaf_nodes=31,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self._config.random_seed,
        )
        return model.fit(x, y)

    def _fit_lr(self, x: np.ndarray, y: np.ndarray) -> _Logistic:
        """Standardized, class-balanced logistic regression (same recipe as the `full` baseline)."""
        scaler = StandardScaler()
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self._config.random_seed)
        model.fit(scaler.fit_transform(x), y)
        return _Logistic(scaler, model)


@dataclass(frozen=True)
class _Logistic:
    """A standardizer plus a fitted logistic regression; scores are logits (never saturate into ties)."""

    scaler: StandardScaler
    model: LogisticRegression

    def score(self, x: np.ndarray) -> np.ndarray:
        """Logit of the positive class."""
        return self.model.decision_function(self.scaler.transform(x))
