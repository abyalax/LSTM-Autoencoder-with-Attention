"""Dense (feed-forward) autoencoder baseline: reconstruction error on single-transaction features."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from lstm_autoencoder.model.config import DenseAutoencoderConfig


class DenseAutoencoder(nn.Module):
    """Two-layer MLP encoder/decoder with a latent bottleneck narrower than the input."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        """Create the encoder and decoder stacks."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct `x` through the latent bottleneck."""
        return self.decoder(self.encoder(x))

    @staticmethod
    def reconstruction_error(x: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Return the per-row mean squared reconstruction error."""
        return ((x - reconstructed) ** 2).mean(dim=1)


@dataclass
class DenseTrainResult:
    """JSON-friendly outcome of one dense autoencoder training run."""

    best_epoch: int
    best_val_loss: float
    history: list[dict[str, float]]


class DenseAutoencoderTrainer:
    """Fit a DenseAutoencoder on normal rows; select the epoch by validation reconstruction loss."""

    def __init__(self, config: DenseAutoencoderConfig) -> None:
        """Bind the trainer to its hyperparameters and pick the device."""
        self._config = config
        self._device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    def fit(
        self, train_x: np.ndarray, val_x: np.ndarray, seed: int
    ) -> tuple[DenseAutoencoder, DenseTrainResult]:
        """Train on normal-only `train_x`; early-stop on the mean loss over normal-only `val_x`."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        config = self._config
        model = DenseAutoencoder(train_x.shape[1], config.hidden_dim, config.latent_dim).to(self._device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        train = torch.as_tensor(train_x, dtype=torch.float32).to(self._device)
        generator = torch.Generator().manual_seed(seed)

        best_loss, best_epoch, best_state, stale = float("inf"), -1, None, 0
        history: list[dict[str, float]] = []
        for epoch in range(1, config.max_epochs + 1):
            model.train()
            order = torch.randperm(len(train), generator=generator).to(self._device)
            total = 0.0
            for start in range(0, len(train), config.batch_size):
                batch = train[order[start : start + config.batch_size]]
                optimizer.zero_grad()
                loss = model.reconstruction_error(batch, model(batch)).mean()
                loss.backward()
                optimizer.step()
                total += loss.detach().item() * len(batch)
            train_loss = total / len(train)
            val_loss = float(self.predict_errors(model, val_x).mean())
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best_loss:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= config.patience:
                    break

        model.load_state_dict(best_state)
        return model, DenseTrainResult(best_epoch=best_epoch, best_val_loss=best_loss, history=history)

    @torch.no_grad()
    def predict_errors(self, model: DenseAutoencoder, x: np.ndarray) -> np.ndarray:
        """Return the per-row reconstruction error, in input order."""
        model.eval()
        tensor = torch.as_tensor(x, dtype=torch.float32)
        step = self._config.batch_size * 4
        errors = []
        for start in range(0, len(tensor), step):
            batch = tensor[start : start + step].to(self._device)
            errors.append(model.reconstruction_error(batch, model(batch)).cpu().numpy())
        return np.concatenate(errors)
