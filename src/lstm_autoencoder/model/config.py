"""Configuration for sequence building and LSTM-Autoencoder training."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_HIDDEN_DIM = 64
SELECTION_METRICS: tuple[str, ...] = ("val_loss", "val_auc")
# Anomaly-score definitions over a window: `mean` over real steps, or `last` = current transaction only.
SCORE_STRATEGIES: tuple[str, ...] = ("mean", "last")


def compose_architecture_tag(base: str, hidden_dim: int, bottleneck_dim: int | None) -> str:
    """Append capacity suffixes to a base architecture tag.

    `_z{n}` marks a latent code of n dimensions and `_h{n}` a non-default hidden
    size, so checkpoints of different capacities never share a name. Shared by
    `ModelConfig` and `LSTMAutoencoder` so the two can never disagree.
    """
    tag = base
    if bottleneck_dim is not None:
        tag += f"_z{bottleneck_dim}"
    if hidden_dim != DEFAULT_HIDDEN_DIM:
        tag += f"_h{hidden_dim}"
    return tag


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
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_layers: int = 2
    attention: bool = True
    bottleneck: bool = False
    attention_pooling: bool = False
    dropout: float = 0.1
    bottleneck_dim: int | None = None

    def __post_init__(self) -> None:
        """Reject capacity settings that would silently build a different model."""
        if self.bottleneck_dim is None:
            return
        if self.bottleneck_dim < 1:
            raise ValueError("bottleneck_dim must be a positive integer.")
        if not (self.bottleneck or self.attention_pooling):
            raise ValueError(
                "bottleneck_dim needs a single-vector architecture: use --bottleneck or "
                "--attention-pooling (plain/attention keep a per-step context that bypasses the code)."
            )

    @property
    def architecture_tag(self) -> str:
        """Short architecture label used in checkpoint names and reports.

        Precedence mirrors `LSTMAutoencoder`: bottleneck > attention pooling >
        self-attention > plain, followed by capacity suffixes (`_z{bottleneck_dim}`,
        `_h{hidden_dim}` when not the default). Single source of truth for every
        module that needs to name or locate a checkpoint.
        """
        if self.bottleneck:
            base = "bottleneck"
        elif self.attention_pooling:
            base = "attn_pool"
        else:
            base = "attn" if self.attention else "plain"
        return compose_architecture_tag(base, self.hidden_dim, self.bottleneck_dim)


@dataclass(frozen=True)
class TrainConfig:
    """Training loop settings, sized for a RTX 3050 6GB-VRAM GPU.

    `selection_metric` picks the checkpoint: `val_loss` is the mean reconstruction
    error of NORMAL validation rows (label-free); `val_auc` is the reconstruction-error
    AUC on labelled validation rows (uses labels, so it is a selection bias that must
    be stated). `min_delta` is the RELATIVE improvement an epoch must show to count as
    progress, so creeping gains below it stop training instead of resetting patience.
    """

    batch_size: int = 16384
    num_workers: int = 8
    max_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5
    device: str = "cuda"
    checkpoint_name: str = "lstm_ae_best.pt"
    seed: int = 42
    selection_metric: str = "val_loss"
    min_delta: float = 0.01
    # P1 "Lengkapi metrik LSTM-AE": standalone test-set reports now carry
    # Recall@FPR alongside AUC/AP for every architecture x window configuration,
    # not only for the hybrid classifier (same targets as `HybridClassifierConfig`).
    fpr_targets: tuple[float, ...] = (0.01, 0.001)

    def __post_init__(self) -> None:
        """Validate the checkpoint-selection settings."""
        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}, got {self.selection_metric!r}.")
        if self.min_delta < 0:
            raise ValueError("min_delta must be >= 0.")


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
