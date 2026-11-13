"""Score trained BankSim LSTM-AE checkpoints: standalone, and on top of the GBM baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.evaluation.banksim.baseline import BaselineEvaluator, FullFeatureData
from lstm_autoencoder.evaluation.banksim.hybrid_scoring import (
    COMPARISONS,
    HybridInputs,
    HybridScorer,
    HybridScoringConfig,
    SplitArrays,
)
from lstm_autoencoder.evaluation.banksim.split import BANKSIM_SPLIT_CONFIG
from lstm_autoencoder.evaluation.metrics import RankingMetrics
from lstm_autoencoder.model.config import SCORE_STRATEGIES, ModelConfig, SequenceConfig, TrainConfig
from lstm_autoencoder.model.trainer import Trainer
from lstm_autoencoder.pipeline.banksim.banksim_pipeline import BankSimModelTrainingPipeline, SplitScoringData
from lstm_autoencoder.reporting.reporter import ReportWritter

PRIMARY = "hybrid_gbm_val_vs_gbm_val_control"
VARIANT_ORDER = (
    "lr_train",
    "gbm_train",
    "gbm_val_control",
    "hybrid_gbm_val",
    "stack_ae",
    "null_hybrid_gbm_val",
    "null_stack",
    "ae_only",
)


class BankSimModelEvaluator:
    """Load each seed's checkpoint (no retraining), score val/test, and compare against GBM.

    Pass the SAME window and architecture flags the checkpoints were trained with: they
    select the checkpoint file. The anomaly-score definition (`mean` over the window vs
    `last` = current transaction only) is picked once on validation AP, averaged over
    seeds, and then used for every seed and for test; the choice uses validation labels
    and is reported as such.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        sequence_config: SequenceConfig,
        model_config: ModelConfig,
        seeds: tuple[int, ...] = (42, 123, 2026),
        n_bootstrap: int = 1000,
        reporter: ReportWritter | None = None,
    ) -> None:
        """Bind the evaluator to the checkpoints' configuration."""
        self._dataset_config = dataset_config
        self._sequence_config = sequence_config
        self._model_config = model_config
        self._seeds = seeds
        self._n_bootstrap = n_bootstrap
        self._reporter = reporter or ReportWritter(dataset_config)
        self._reporter.set_path_info_from_module()

    def run(self) -> dict[str, object]:
        """Score every seed, write the report, and return the aggregate."""
        pipeline = BankSimModelTrainingPipeline(
            dataset_config=self._dataset_config,
            sequence_config=self._sequence_config,
            model_config=self._model_config,
            split_config=BANKSIM_SPLIT_CONFIG,
        )
        scoring = pipeline.prepared_scoring_splits()  # also fixes feature columns / input_dim
        data = BaselineEvaluator(
            dataset_config=self._dataset_config, split_config=BANKSIM_SPLIT_CONFIG, keep_account_code=False
        ).prepare_full_data()
        inputs = self._aligned_inputs(data, scoring)
        self._reporter.display_text(
            f"train {len(inputs.train.y):,} | val {len(inputs.val.y):,} ({int(inputs.val.y.sum()):,} fraud) | "
            f"test {len(inputs.test.y):,} ({int(inputs.test.y.sum()):,} fraud) | tabular features {inputs.test.x.shape[1]}"
        )

        errors = {seed: self._score_errors(pipeline, scoring, seed) for seed in self._seeds}
        standalone = self._standalone(errors, inputs)
        strategy = str(standalone["selected_strategy"])

        scorer = HybridScorer(inputs, HybridScoringConfig(n_bootstrap=self._n_bootstrap))
        per_seed = [
            scorer.score_seed(seed, {name: errors[seed][name][strategy] for name in ("val", "test")})
            for seed in self._seeds
        ]
        aggregate = self._aggregate(per_seed)
        self._report(standalone, per_seed, aggregate, pipeline)
        self._reporter.update_summary(
            {
                "banksim_model_evaluation": {
                    "window_length": pipeline.sequence_config.window_length,
                    "architecture": pipeline.model_config.architecture_tag,
                    "feature_columns": list(pipeline.sequence_config.feature_columns),
                    "seeds": list(self._seeds),
                    "n_bootstrap": self._n_bootstrap,
                    "standalone": standalone,
                    "per_seed": per_seed,
                    "aggregate": aggregate,
                }
            }
        )
        self._reporter.save_reports(
            text_filename="banksim_model_evaluation.txt", json_filename="banksim_model_evaluation.json"
        )
        return aggregate

    @staticmethod
    def _aligned_inputs(data: FullFeatureData, scoring: dict[str, SplitScoringData]) -> HybridInputs:
        """Pair each split's AE windows with the tabular rows of the SAME transactions, or fail loudly."""
        matrix = data.features.to_numpy(dtype="float64")
        splits: dict[str, SplitArrays] = {}
        for name, split in scoring.items():
            rows = np.asarray(split.row_index)
            if not np.array_equal(np.sort(data.splits[name]), rows):
                raise RuntimeError(f"Baseline and LSTM-AE disagree on the {name} rows; the splits are not shared.")
            if not np.array_equal(data.labels[rows], np.asarray(split.labels)):
                raise RuntimeError(f"Baseline and LSTM-AE disagree on the {name} labels.")
            splits[name] = SplitArrays(matrix[rows], data.labels[rows], data.account_code[rows])
        return HybridInputs(train=splits["train"], val=splits["val"], test=splits["test"])

    def _score_errors(
        self, pipeline: BankSimModelTrainingPipeline, scoring: dict[str, SplitScoringData], seed: int
    ) -> dict[str, dict[str, np.ndarray]]:
        """{split: {strategy: per-row error}} for val and test, from this seed's checkpoint."""
        checkpoint = pipeline.checkpoint_path(seed)
        if not checkpoint.exists():
            raise ValueError(
                f"Checkpoint not found: {checkpoint}. Train it first with `train-model --dataset banksim "
                f"--window {self._sequence_config.window_length} --seeds {seed}` and the same architecture flags."
            )
        trainer = Trainer(pipeline.model_config, TrainConfig(seed=seed), checkpoint.parent)
        model = trainer.build_eval_model(checkpoint)
        return {
            name: {strategy: trainer.predict_errors(model, scoring[name].dataset, strategy) for strategy in SCORE_STRATEGIES}
            for name in ("val", "test")
        }

    def _standalone(
        self, errors: dict[int, dict[str, dict[str, np.ndarray]]], inputs: HybridInputs
    ) -> dict[str, object]:
        """AE-only AUC/AP per seed and strategy; pick the strategy by mean validation AP."""
        table: dict[str, dict[str, list[dict[str, float]]]] = {s: {"val": [], "test": []} for s in SCORE_STRATEGIES}
        for seed_errors in errors.values():
            for strategy in SCORE_STRATEGIES:
                for name, split in (("val", inputs.val), ("test", inputs.test)):
                    scores = seed_errors[name][strategy]
                    table[strategy][name].append(
                        {
                            "auc": float(RankingMetrics.roc_auc(scores, split.y)),
                            "ap": float(RankingMetrics.average_precision(scores, split.y)),
                        }
                    )
        val_ap = {s: float(np.mean([m["ap"] for m in table[s]["val"]])) for s in SCORE_STRATEGIES}
        selected = max(val_ap, key=lambda strategy: val_ap[strategy])
        return {"by_strategy": table, "val_ap_mean": val_ap, "selected_strategy": selected}

    @staticmethod
    def _aggregate(per_seed: list[dict[str, object]]) -> dict[str, object]:
        """Mean and std across seeds of every variant metric and every delta."""

        def collect(key: str) -> dict[str, dict[str, dict[str, float]]]:
            names = per_seed[0][key].keys()  # type: ignore[union-attr]
            out: dict[str, dict[str, dict[str, float]]] = {}
            for name in names:
                metrics = per_seed[0][key][name].keys()  # type: ignore[index, union-attr]
                out[name] = {}
                for metric in metrics:
                    values = np.array([run[key][name][metric] for run in per_seed], dtype="float64")  # type: ignore[index]
                    out[name][metric] = {
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    }
            return out

        return {"variants": collect("variants"), "deltas": collect("deltas")}

    def _report(
        self,
        standalone: dict[str, object],
        per_seed: list[dict[str, object]],
        aggregate: dict[str, object],
        pipeline: BankSimModelTrainingPipeline,
    ) -> None:
        """Console/text report: standalone AE, the variant table, and the primary deltas with their CIs."""
        reporter = self._reporter
        reporter.display_section(
            f"BankSim LSTM-AE evaluation (W={pipeline.sequence_config.window_length}, "
            f"{pipeline.model_config.architecture_tag}, seeds={list(self._seeds)})"
        )
        by_strategy = standalone["by_strategy"]  # type: ignore[assignment]
        for strategy in SCORE_STRATEGIES:
            val = by_strategy[strategy]["val"]  # type: ignore[index]
            test = by_strategy[strategy]["test"]  # type: ignore[index]
            marker = "  <- selected on val AP" if strategy == standalone["selected_strategy"] else ""
            reporter.display_text(
                f"AE only, score={strategy}: val AUC {np.mean([m['auc'] for m in val]):.4f} / "
                f"AP {np.mean([m['ap'] for m in val]):.4f} | test AUC {np.mean([m['auc'] for m in test]):.4f} / "
                f"AP {np.mean([m['ap'] for m in test]):.4f}{marker}"
            )

        variants = aggregate["variants"]  # type: ignore[assignment]
        rows = []
        for name in VARIANT_ORDER:
            entry = variants[name]  # type: ignore[index]
            rows.append(
                {
                    "variant": name,
                    "auc": f"{entry['auc']['mean']:.4f} +/- {entry['auc']['std']:.4f}",
                    "ap": f"{entry['ap']['mean']:.4f} +/- {entry['ap']['std']:.4f}",
                    "recall@1%fpr": f"{entry['recall_at_fpr_0.01']['mean']:.4f}",
                    "recall@0.1%fpr": f"{entry['recall_at_fpr_0.001']['mean']:.4f}",
                }
            )
        reporter.display_section("Test metrics, mean +/- std over seeds")
        reporter.display_table(pd.DataFrame(rows))

        reporter.display_section("Does the AE error add anything? (test, customer-cluster bootstrap 95% CI per seed)")
        for candidate, reference in COMPARISONS:
            pair = f"{candidate}_vs_{reference}"
            delta = aggregate["deltas"][pair]  # type: ignore[index]
            reporter.display_text(
                f"{pair}: dAUC {delta['auc']['mean']:+.4f} +/- {delta['auc']['std']:.4f} | "
                f"dAP {delta['ap']['mean']:+.4f} +/- {delta['ap']['std']:.4f}"
            )
            if self._n_bootstrap > 0:
                for run in per_seed:
                    ci = run["bootstrap"][pair]  # type: ignore[index]
                    reporter.display_text(
                        f"    seed {run['seed']}: dAUC [{ci['d_auc_ci_low']:+.4f}, {ci['d_auc_ci_high']:+.4f}] | "
                        f"dAP [{ci['d_ap_ci_low']:+.4f}, {ci['d_ap_ci_high']:+.4f}]"
                    )
        reporter.display_text(
            f"Primary comparison: {PRIMARY}. A gain counts only if its CI excludes 0 in every seed and "
            "the null_* rows stay near 0."
        )
