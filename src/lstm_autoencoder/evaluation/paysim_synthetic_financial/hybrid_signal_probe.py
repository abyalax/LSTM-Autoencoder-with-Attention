"""Check whether LSTM-AE reconstruction error adds signal beyond the LR baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.evaluation.metrics import RankingMetrics
from lstm_autoencoder.evaluation.paysim_synthetic_financial.baseline import BaselineEvaluator
from lstm_autoencoder.model.config import ModelConfig, SequenceConfig, TrainConfig
from lstm_autoencoder.model.pipeline import PaySimModelTrainingPipeline
from lstm_autoencoder.model.trainer import Trainer
from lstm_autoencoder.reporting.reporter import ReportWritter


@dataclass(frozen=True)
class HybridSignalResult:
    """JSON-friendly outcome of the correlation probe."""

    n_test: int
    lr_test_auc: float
    ae_test_auc: float
    ae_test_ap: float
    pearson_r: float
    spearman_r: float
    mean_ae_error_by_lr_outcome: dict[str, float]


class HybridSignalProbe:
    """Correlate a trained LSTM-AE's reconstruction error with the LR baseline's
    score, on the SAME test rows, before committing to a full hybrid retrain.

    Loads an already-trained checkpoint (no retraining) and refits the LR
    'full' baseline (seconds, not a GPU job). Low correlation between the two
    signals suggests the reconstruction error could add complementary signal
    to the LR; high correlation suggests it is mostly redundant with what the
    LR already captures from the same underlying features.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        sequence_config: SequenceConfig | None = None,
        model_config: ModelConfig | None = None,
        checkpoint_path: Path | None = None,
        seed: int = 42,
        reporter: ReportWritter | None = None,
    ) -> None:
        """Bind the probe to its configuration and the checkpoint to load."""
        self._dataset_config = dataset_config
        self._sequence_config = sequence_config or SequenceConfig()
        self._model_config = model_config or ModelConfig()
        self._seed = seed
        self._checkpoint_path = checkpoint_path or self._default_checkpoint_path()
        self._reporter = reporter or ReportWritter(dataset_config)
        self._reporter.set_path_info_from_module()

    def run(self) -> HybridSignalResult:
        """Score the LSTM-AE checkpoint and the LR baseline on the same test
        rows, then report how correlated the two signals are."""
        ae_test_idx, ae_errors, ae_labels = self._score_autoencoder()
        lr_test_idx, lr_scores, lr_labels = self._score_baseline()
        self._assert_aligned(ae_test_idx, lr_test_idx, ae_labels, lr_labels)

        ae_auc, ae_ap = RankingMetrics.roc_auc(ae_errors, ae_labels), RankingMetrics.average_precision(
            ae_errors, ae_labels
        )
        lr_auc = RankingMetrics.roc_auc(lr_scores, lr_labels)

        result = HybridSignalResult(
            n_test=len(ae_errors),
            lr_test_auc=lr_auc,
            ae_test_auc=ae_auc,
            ae_test_ap=ae_ap,
            pearson_r=self._pearson(ae_errors, lr_scores),
            spearman_r=self._pearson(self._rank(ae_errors), self._rank(lr_scores)),
            mean_ae_error_by_lr_outcome=self._mean_error_by_lr_outcome(ae_errors, lr_scores, lr_labels),
        )
        self._report(result)
        return result

    def _score_autoencoder(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load the trained checkpoint (no retraining) and score the test windows."""
        pipeline = PaySimModelTrainingPipeline(
            dataset_config=self._dataset_config,
            sequence_config=self._sequence_config,
            model_config=self._model_config,
            # NOT self._seed: the receiver train/val/test split always used the
            # pipeline's default (42) during training, regardless of which torch
            # seed trained a given checkpoint -- only the checkpoint filename varies.
        )
        test_subset, test_labels, test_row_index = pipeline.prepared_test_data()

        trainer = Trainer(self._model_config, TrainConfig(), self._checkpoint_path.parent)
        model = trainer.build_eval_model(self._checkpoint_path)
        errors = trainer.predict_errors(model, test_subset)
        return test_row_index, errors, test_labels

    def _score_baseline(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit the LR 'full' baseline fresh and score the test rows."""
        return BaselineEvaluator(config=self._dataset_config).fit_full_variant()

    def _default_checkpoint_path(self) -> Path:
        """Checkpoint path matching PaySimModelTrainingPipeline's naming
        convention for the current window, architecture, and seed."""
        if self._model_config.bottleneck:
            tag = "bottleneck"
        else:
            tag = "attn" if self._model_config.attention else "plain"
        name = f"lstm_ae_w{self._sequence_config.window_length}_{tag}_seed{self._seed}_best.pt"
        return self._dataset_config.artifacts_dir / "model" / "paysim_synthetic_financial" / name

    @staticmethod
    def _assert_aligned(
        ae_test_idx: np.ndarray, lr_test_idx: np.ndarray, ae_labels: np.ndarray, lr_labels: np.ndarray
    ) -> None:
        """Fail loudly if the two pipelines' test rows don't line up -- any
        correlation below is meaningless unless both score the SAME rows."""
        if not np.array_equal(ae_test_idx, lr_test_idx):
            raise ValueError(
                "LSTM-AE and LR test row positions differ -- check both use the "
                "same --dataset, --nrows, and seed (default 42) for the receiver split."
            )
        if not np.array_equal(ae_labels, lr_labels):
            raise ValueError("Labels differ between LSTM-AE and LR test rows despite matching positions.")

    @staticmethod
    def _mean_error_by_lr_outcome(errors: np.ndarray, lr_scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """Mean AE reconstruction error split by label, and -- for fraud rows --
        by whether the LR scored it low (LR's likely blind spot) or high (LR
        already confident). A higher error on 'fraud_lr_low' than on
        'fraud_lr_high' or 'normal' is evidence of complementary signal."""
        fraud = labels == 1
        normal_mean = float(errors[~fraud].mean()) if (~fraud).any() else float("nan")
        if not fraud.any():
            return {"normal": normal_mean, "fraud_lr_low": float("nan"), "fraud_lr_high": float("nan")}

        fraud_scores, fraud_errors = lr_scores[fraud], errors[fraud]
        low = fraud_scores <= np.median(fraud_scores)
        return {
            "normal": normal_mean,
            "fraud_lr_low": float(fraud_errors[low].mean()),
            "fraud_lr_high": float(fraud_errors[~low].mean()),
        }

    @staticmethod
    def _rank(values: np.ndarray) -> np.ndarray:
        """Average ranks (ties split evenly), the basis for Spearman correlation."""
        return pd.Series(values).rank().to_numpy()

    @staticmethod
    def _pearson(x: np.ndarray, y: np.ndarray) -> float:
        """Pearson correlation coefficient between two equal-length arrays."""
        return float(np.corrcoef(x, y)[0, 1])

    def _report(self, result: HybridSignalResult) -> None:
        """Print and save the correlation probe's findings."""
        self._reporter.display_section("Hybrid signal probe: LSTM-AE reconstruction error vs LR baseline score")
        self._reporter.display_text(
            f"n_test {result.n_test:,} | LR test AUC {result.lr_test_auc:.4f} | "
            f"AE test AUC {result.ae_test_auc:.4f} (AP {result.ae_test_ap:.4f})"
        )
        self._reporter.display_text(
            f"Pearson r {result.pearson_r:.4f} | Spearman r {result.spearman_r:.4f} "
            "(near 0 = complementary signal, near 1 = redundant with LR)"
        )
        self._reporter.display_table(pd.DataFrame({"mean_ae_error": result.mean_ae_error_by_lr_outcome}))
        self._reporter.update_summary({"hybrid_signal_probe": asdict(result)})
        self._reporter.save_reports(
            text_filename="hybrid_signal_probe_report.txt",
            json_filename="hybrid_signal_probe_report.json",
        )