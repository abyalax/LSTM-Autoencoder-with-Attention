"""Training loop for the LSTM-Autoencoder, with a configurable checkpoint-selection metric."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lstm_autoencoder.evaluation.metrics import RankingMetrics
from lstm_autoencoder.model.config import ModelConfig, TrainConfig
from lstm_autoencoder.model.lstm_autoencoder import LSTMAutoencoder


@dataclass
class TrainResult:
    """JSON-friendly outcome of one training run."""

    best_epoch: int
    best_val_loss: float
    best_val_auc: float
    peak_val_auc: float
    peak_val_auc_epoch: int
    selection_metric: str
    history: list[dict[str, float]]
    checkpoint_path: str


class Trainer:
    """Fit an LSTMAutoencoder on normal-only windows and keep the best checkpoint.

    The checkpoint is chosen by `TrainConfig.selection_metric`; the validation
    AUC at that checkpoint and the best AUC seen at ANY epoch are both recorded,
    so a gap between "selected" and "peak" is visible in every report.
    """

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig, checkpoint_dir: Path) -> None:
        """Bind the trainer to model and training hyperparameters."""
        self._model_config = model_config
        self._train_config = train_config
        self._checkpoint_dir = checkpoint_dir
        self._device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        val_labels: np.ndarray
    ) -> tuple[LSTMAutoencoder, TrainResult]:
        """Train on normal-only windows; select the checkpoint by `selection_metric`."""
        torch.manual_seed(self._train_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._train_config.seed)
        model = self._build_model()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self._train_config.learning_rate,
            weight_decay=self._train_config.weight_decay,
        )
        train_loader = self._make_loader(train_dataset, shuffle=True)
        val_loader = self._make_loader(val_dataset, shuffle=False)

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self._checkpoint_dir / self._train_config.checkpoint_name

        selection = self._train_config.selection_metric
        best_score, best_epoch, stale_epochs = float("inf"), -1, 0
        history: list[dict[str, float]] = []

        for epoch in range(1, self._train_config.max_epochs + 1):
            train_loss = self._train_epoch(model, train_loader, optimizer)
            val_errors = self._predict_errors(model, val_loader)
            val_auc = float(RankingMetrics.roc_auc(val_errors, val_labels))
            normal = val_labels == 0
            val_loss = float(val_errors[normal].mean() if normal.any() else val_errors.mean())
            history.append({"epoch": epoch, "train_loss": float(train_loss), "val_loss": val_loss, "val_auc": val_auc})

            # Both metrics are minimised: AUC is negated so "higher is better" works too.
            score = val_loss if selection == "val_loss" else -val_auc
            if not np.isfinite(score):
                raise ValueError(f"Selection metric '{selection}' is not finite at epoch {epoch}: {score}")

            if self._improved(score, best_score):
                best_score, best_epoch, stale_epochs = score, epoch, 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                stale_epochs += 1
                if stale_epochs >= self._train_config.patience:
                    break

        model.load_state_dict(torch.load(checkpoint_path, map_location=self._device))
        selected = history[best_epoch - 1]  # epochs are 1-based and consecutive
        peak = max(history, key=lambda entry: -1.0 if np.isnan(entry["val_auc"]) else entry["val_auc"])
        result = TrainResult(
            best_epoch=best_epoch,
            best_val_loss=selected["val_loss"],
            best_val_auc=selected["val_auc"],
            peak_val_auc=peak["val_auc"],
            peak_val_auc_epoch=int(peak["epoch"]),
            selection_metric=selection,
            history=history,
            checkpoint_path=str(checkpoint_path),
        )

        del train_loader
        del val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model, result

    def score(self, model: LSTMAutoencoder, dataset: Dataset, labels: np.ndarray) -> dict[str, float]:
        """Return AUC, average precision, and recall@FPR (per `TrainConfig.fpr_targets`)
        of reconstruction error against labels.

        Was a bare `(auc, ap)` tuple; P1 "Lengkapi metrik LSTM-AE" extends every
        standalone test-set report with the same Recall@FPR targets the hybrid
        classifier already reports, so the two are comparable metric-for-metric.
        """
        return self._score_errors(self.predict_errors(model, dataset), labels)

    def predict_errors(self, model: LSTMAutoencoder, dataset: Dataset) -> np.ndarray:
        """Run inference and return per-example reconstruction error, in dataset order."""
        return self._predict_errors(model, self._make_loader(dataset, shuffle=False))

    def predict_step_errors(self, model: LSTMAutoencoder, dataset: Dataset) -> np.ndarray:
        """Return the per-step reconstruction error, shape (n, window), in dataset order.

        Padded steps are 0. Windows are left-padded, so the LAST column is always the
        current transaction; the mean over real steps equals `predict_errors`.
        """
        return self._predict_step_errors(model, self._make_loader(dataset, shuffle=False))

    def _improved(self, score: float, best: float) -> bool:
        """True when `score` beats `best` by at least the relative `min_delta` (first epoch always counts)."""
        if not np.isfinite(best):
            return True
        return score < best - abs(best) * self._train_config.min_delta

    def build_eval_model(self, checkpoint_path: Path) -> LSTMAutoencoder:
        """Build the configured architecture and load trained weights, for
        inference only (no optimizer, no training loop)."""
        model = self._build_model()
        model.load_state_dict(torch.load(checkpoint_path, map_location=self._device))
        model.eval()
        return model

    def _score_errors(self, errors: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """AUC, AP and recall@FPR of a reconstruction-error array against labels."""
        out = {
            "auc": float(RankingMetrics.roc_auc(errors, labels)),
            "ap": float(RankingMetrics.average_precision(errors, labels)),
        }
        for fpr in self._train_config.fpr_targets:
            out[f"recall_at_fpr_{fpr:g}"] = float(RankingMetrics.recall_at_fpr(errors, labels, fpr))
        return out
    
    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """Build a DataLoader with worker/pinning settings tuned for an under-used GPU."""
        return DataLoader(
            dataset,
            batch_size=self._train_config.batch_size,
            shuffle=shuffle,
            num_workers=self._train_config.num_workers,
            pin_memory=self._device.type == "cuda",
            persistent_workers=self._train_config.num_workers > 0,
        )

    def _build_model(self) -> LSTMAutoencoder:
        """Instantiate the model on the trainer's device.

        Raises if the built model's architecture differs from the requested
        config, so a flag that is not forwarded can never silently train (and
        get a checkpoint named after) a different architecture.
        """
        model = LSTMAutoencoder(
            input_dim=self._model_config.input_dim,
            hidden_dim=self._model_config.hidden_dim,
            num_layers=self._model_config.num_layers,
            attention=self._model_config.attention,
            bottleneck=self._model_config.bottleneck,
            attention_pooling=self._model_config.attention_pooling,
            dropout=self._model_config.dropout,
            bottleneck_dim=self._model_config.bottleneck_dim,
        )
        if model.architecture_tag != self._model_config.architecture_tag:
            raise RuntimeError(
                f"Architecture mismatch: config requests '{self._model_config.architecture_tag}' "
                f"but the built model is '{model.architecture_tag}'."
            )
        return model.to(self._device)

    def _train_epoch(self, model: LSTMAutoencoder, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        """Run one training epoch and return the mean reconstruction loss."""
        model.train()
        total_loss, total_examples = 0.0, 0
        for sequence, mask, _ in loader:
            sequence = sequence.to(self._device, non_blocking=True)
            mask = mask.to(self._device, non_blocking=True)
            optimizer.zero_grad()
            reconstructed = model(sequence, mask)
            loss = model.reconstruction_error(sequence, reconstructed, mask).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * len(sequence)
            total_examples += len(sequence)
        return total_loss / max(total_examples, 1)

    @torch.no_grad()
    def _predict_step_errors(self, model: LSTMAutoencoder, loader: DataLoader) -> np.ndarray:
        """Run inference and return per-step reconstruction error for every example."""
        model.eval()
        chunks = []
        for sequence, mask, _ in loader:
            sequence = sequence.to(self._device, non_blocking=True)
            mask = mask.to(self._device, non_blocking=True)
            reconstructed = model(sequence, mask)
            chunks.append(model.stepwise_error(sequence, reconstructed, mask).cpu().numpy())
        return np.concatenate(chunks)

    @torch.no_grad()
    def _predict_errors(self, model: LSTMAutoencoder, loader: DataLoader) -> np.ndarray:
        """Run inference and return per-example reconstruction error."""
        model.eval()
        errors = []
        for sequence, mask, _ in loader:
            sequence = sequence.to(self._device, non_blocking=True)
            mask = mask.to(self._device, non_blocking=True)
            reconstructed = model(sequence, mask)
            errors.append(model.reconstruction_error(sequence, reconstructed, mask).cpu().numpy())
        return np.concatenate(errors)