"""Training loop for the LSTM-Autoencoder, with early stopping on validation AUC."""

from __future__ import annotations

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
    best_val_auc: float
    history: list[dict[str, float]]
    checkpoint_path: str


class Trainer:
    """Fit an LSTMAutoencoder on normal-only windows, select by validation AUC."""

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
        """Train on normal-only windows; validate using reconstruction-error AUC."""
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

        best_auc, best_epoch, stale_epochs = -1.0, -1, 0
        history: list[dict[str, float]] = []

        for epoch in range(1, self._train_config.max_epochs + 1):
            train_loss = self._train_epoch(model, train_loader, optimizer)
            val_errors = self._predict_errors(model, val_loader)
            val_auc = RankingMetrics.roc_auc(val_errors, val_labels)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc})

            if val_auc > best_auc:
                best_auc, best_epoch, stale_epochs = val_auc, epoch, 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                stale_epochs += 1
                if stale_epochs >= self._train_config.patience:
                    break

        model.load_state_dict(torch.load(checkpoint_path, map_location=self._device))
        result = TrainResult(
            best_epoch=best_epoch,
            best_val_auc=best_auc,
            history=history,
            checkpoint_path=str(checkpoint_path),
        )
        return model, result

    def score(self, model: LSTMAutoencoder, dataset: Dataset, labels: np.ndarray) -> tuple[float, float]:
        """Return (AUC, average precision) of reconstruction error against labels."""
        return self._score_errors(self.predict_errors(model, dataset), labels)

    def predict_errors(self, model: LSTMAutoencoder, dataset: Dataset) -> np.ndarray:
        """Run inference and return per-example reconstruction error, in dataset order."""
        return self._predict_errors(model, self._make_loader(dataset, shuffle=False))

    def build_eval_model(self, checkpoint_path: Path) -> LSTMAutoencoder:
        """Build the configured architecture and load trained weights, for
        inference only (no optimizer, no training loop)."""
        model = self._build_model()
        model.load_state_dict(torch.load(checkpoint_path, map_location=self._device))
        model.eval()
        return model

    @staticmethod
    def _score_errors(errors: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
        """AUC and average precision of a reconstruction-error array against labels."""
        return RankingMetrics.roc_auc(errors, labels), RankingMetrics.average_precision(errors, labels)
    
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
        """Instantiate the model on the trainer's device."""
        return LSTMAutoencoder(
            input_dim=self._model_config.input_dim,
            hidden_dim=self._model_config.hidden_dim,
            num_layers=self._model_config.num_layers,
            attention=self._model_config.attention,
            bottleneck=self._model_config.bottleneck,
            dropout=self._model_config.dropout,
        ).to(self._device)

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