"""Configuration for sequence building and LSTM-Autoencoder training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceConfig:
    """Settings for turning per-receiver transactions into fixed-length windows.
    
    These properties are determined during exploratory data analysis and
    should be configured based on exploration results.
    """

    window_length: int = 5
    feature_columns: tuple[str, ...] = (
        "amount",
        "step_diff",
        "log_amount_ratio",
        "history_length",
        "is_new_receiver",
    )
    min_history_length: int = 1


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the LSTM-Autoencoder with attention."""

    input_dim: int = len(SequenceConfig().feature_columns)
    hidden_dim: int = 64
    num_layers: int = 2
    attention: bool = True
    bottleneck: bool = False
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    """Training loop settings, sized for a 6GB-VRAM GPU."""

    batch_size: int = 4096
    num_workers: int = 4
    max_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5
    device: str = "cuda"
    checkpoint_name: str = "lstm_ae_best.pt"
    seed: int = 42


@dataclass(frozen=True)
class DenseAutoencoderConfig:
    """Hyperparameters for the dense (feed-forward) autoencoder baseline.

    `latent_dim` is deliberately smaller than the number of input features (5),
    otherwise the network can learn the identity and reconstruct every row,
    fraud included. Early stopping uses validation reconstruction loss on
    normal rows only (no labels needed to pick the epoch).
    """

    hidden_dim: int = 32
    latent_dim: int = 3
    batch_size: int = 4096
    max_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5
    device: str = "cuda"
