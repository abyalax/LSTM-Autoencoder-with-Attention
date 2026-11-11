"""Choose the reconstruction-score strategy (mean vs last) on validation and freeze it before test."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from torch.utils.data import Dataset

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.evaluation.config import SplitConfig
from lstm_autoencoder.evaluation.metrics import RankingMetrics
from lstm_autoencoder.evaluation.paysim_synthetic_financial.score_strategy_registry import (
    FrozenScoreStrategy,
    checkpoint_key,
    config_key,
)
from lstm_autoencoder.evaluation.paysim_synthetic_financial.standalone_evaluator import (
    ARCHITECTURES,
    model_config_for,
)
from lstm_autoencoder.model.config import DEFAULT_HIDDEN_DIM, SCORE_STRATEGIES, SequenceConfig, TrainConfig
from lstm_autoencoder.model.pipeline import PaySimModelTrainingPipeline
from lstm_autoencoder.model.trainer import Trainer
from lstm_autoencoder.reporting.reporter import ReportWritter

DEFAULT_TIE_BREAK = "mean"  # the training loss, and the score reported so far


@dataclass(frozen=True)
class ScoreStrategyConfig:
    """Settings of the validation-only score-strategy selection.

    The strategy with the higher mean validation AP across `seeds` wins; a difference
    smaller than `min_gain` (absolute AP) keeps the default `mean`, so a noise-level
    edge on one split never flips the choice. Test labels are never read here.
    """

    seeds: tuple[int, ...] = (42, 123, 2026)
    min_gain: float = 0.0
    force: bool = False


class ScoreStrategySelector:
    """Pick `mean` or `last` per (architecture, window) from VALIDATION AP, then write the frozen registry.

    One forward pass per checkpoint yields both strategies (they are two read-outs of
    the same per-step error). The registry is written once and refuses to be
    overwritten unless `force=True`, so a later look at test numbers cannot quietly
    change the choice.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        windows: tuple[int, ...] = (1, 3, 5, 10),
        architectures: tuple[str, ...] = ARCHITECTURES,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        bottleneck_dim: int | None = None,
        split_config: SplitConfig | None = None,
        config: ScoreStrategyConfig | None = None,
        reporter: ReportWritter | None = None,
    ) -> None:
        """Bind the selector to the grid of configurations to freeze."""
        self._dataset_config = dataset_config
        self._windows = windows
        self._architectures = architectures
        self._hidden_dim = hidden_dim
        self._bottleneck_dim = bottleneck_dim
        self._split_config = split_config or SplitConfig()
        self._config = config or ScoreStrategyConfig()
        self._reporter = reporter or ReportWritter(dataset_config)
        self._reporter.set_path_info_from_module()

    def run(self) -> dict[str, dict[str, object]]:
        """Select a strategy for every configuration, write the registry, and return its entries."""
        path = FrozenScoreStrategy.registry_path(self._dataset_config.artifacts_dir)
        if path.exists() and not self._config.force:
            raise FileExistsError(
                f"{path} already exists: the strategy is frozen. Pass --force only after RETRAINING the checkpoints."
            )
        self._reporter.display_section(f"Score Strategy Selection (validation only, seeds={list(self._config.seeds)})")
        entries: dict[str, dict[str, object]] = {}
        for window in self._windows:
            sequence_config = SequenceConfig(window_length=window)
            first_pipeline = PaySimModelTrainingPipeline(
                dataset_config=self._dataset_config,
                sequence_config=sequence_config,
                model_config=model_config_for(self._architectures[0], self._hidden_dim, self._bottleneck_dim),
                split_config=self._split_config,
            )
            val = first_pipeline.prepared_eval_splits()["val"]
            valid_steps = val.dataset.valid_steps()
            labels = np.asarray(val.labels).astype(int)
            for architecture in self._architectures:
                entry = self._select(architecture, sequence_config, val.dataset, valid_steps, labels)
                key = config_key(str(entry["architecture"]), window)
                entries[key] = entry
                self._reporter.display_text(
                    f"[{key}] val AP mean={entry['val_ap']['mean']:.5f} last={entry['val_ap']['last']:.5f} "  # type: ignore[index]
                    f"-> {entry['strategy']}"
                )

        payload = {
            "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "selection_split": "val",
            "selection_metric": "average_precision",
            "candidates": list(SCORE_STRATEGIES),
            "default_on_tie": DEFAULT_TIE_BREAK,
            "min_gain": self._config.min_gain,
            "seeds": list(self._config.seeds),
            "entries": entries,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._reporter.display_text(f"Frozen registry written to {path}")
        self._reporter.update_summary({"score_strategy": payload})
        self._reporter.save_reports(
            text_filename="score_strategy_report.txt", json_filename="score_strategy_report.json"
        )
        return entries

    @staticmethod
    def choose_strategy(mean_ap: dict[str, float], min_gain: float = 0.0) -> str:
        """`last` only when its mean validation AP beats `mean` by MORE than `min_gain`; otherwise `mean`."""
        if mean_ap["last"] - mean_ap["mean"] > min_gain:
            return "last"
        return DEFAULT_TIE_BREAK

    def _select(
        self,
        architecture: str,
        sequence_config: SequenceConfig,
        dataset: Dataset,
        valid_steps: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, object]:
        """Mean validation AP of both strategies over seeds for one configuration, and the winner."""
        model_config = model_config_for(architecture, self._hidden_dim, self._bottleneck_dim)
        pipeline = PaySimModelTrainingPipeline(
            dataset_config=self._dataset_config,
            sequence_config=sequence_config,
            model_config=model_config,
            split_config=self._split_config,
        )
        per_strategy: dict[str, list[float]] = {name: [] for name in SCORE_STRATEGIES}
        checkpoints: list[str] = []
        for seed in self._config.seeds:
            checkpoint = pipeline.checkpoint_path(seed)
            if not checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Train {model_config.architecture_tag}.")
            trainer = Trainer(model_config, TrainConfig(seed=seed), checkpoint.parent)
            model = trainer.build_eval_model(checkpoint)
            steps = trainer.predict_step_errors(model, dataset)
            scores = {
                "mean": steps.sum(axis=1) / np.maximum(valid_steps, 1),
                "last": steps[np.arange(len(steps)), np.maximum(valid_steps, 1) - 1],
            }
            for name, values in scores.items():
                per_strategy[name].append(float(RankingMetrics.average_precision(values, labels)))
            checkpoints.append(checkpoint_key(checkpoint))

        mean_ap = {name: float(np.mean(values)) for name, values in per_strategy.items()}
        chosen = self.choose_strategy(mean_ap, self._config.min_gain)
        return {
            "architecture": model_config.architecture_tag,
            "window_length": sequence_config.window_length,
            "strategy": chosen,
            "val_ap": mean_ap,
            "val_ap_per_seed": per_strategy,
            "checkpoints": checkpoints,
        }
