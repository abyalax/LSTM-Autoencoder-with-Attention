"""Configuration for sequence building and LSTM-Autoencoder training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceConfig:
    """Settings for turning per-receiver transactions into fixed-length windows."""

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