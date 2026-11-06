"""Hybrid LSTM-Autoencoder with attention for sequence reconstruction."""

from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    """LSTM encoder producing per-step hidden states."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        """Create the encoder LSTM."""
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Return per-step outputs and the final (hidden, cell) state."""
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, (hidden, cell)


class SelfAttention(nn.Module):
    """Scaled dot-product self-attention over encoder outputs, with padding mask."""

    def __init__(self, hidden_dim: int) -> None:
        """Create the query/key/value projections."""
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** 0.5

    def forward(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
       """Attend each step to the OTHER real steps in its window, never to
       itself, unless it is the only real step (then fall back to self)."""
       query = self.query(encoded)
       key = self.key(encoded)
       value = self.value(encoded)

       scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
       padding = (mask == 0).unsqueeze(1)

       window = encoded.size(1)
       diagonal = torch.eye(window, dtype=torch.bool, device=encoded.device).unsqueeze(0)
       real_count = mask.sum(dim=1, keepdim=True)  # (batch, 1)
       exclude_self = diagonal & (real_count > 1).unsqueeze(-1)

       scores = scores.masked_fill(padding | exclude_self, float("-inf"))
       weights = torch.softmax(scores, dim=-1)
       weights = torch.nan_to_num(weights)
       return torch.matmul(weights, value)


class AttentionPooling(nn.Module):
    """Attention-based pooling to compress encoder outputs to a single bottleneck vector."""

    def __init__(self, hidden_dim: int) -> None:
        """Create the attention pooling layer with a learnable query vector."""
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** 0.5

    def forward(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pool encoder outputs to a single vector using attention.
        
        Args:
            encoded: (batch, seq_len, hidden_dim) encoder outputs
            mask: (batch, seq_len) padding mask (1 = real, 0 = padding)
            
        Returns:
            (batch, hidden_dim) pooled representation
        """
        batch_size = encoded.size(0)
        
        # Expand query to batch dimension
        query = self.query.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)  # (batch, 1, hidden_dim)
        
        key = self.key(encoded)  # (batch, seq_len, hidden_dim)
        value = self.value(encoded)  # (batch, seq_len, hidden_dim)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale  # (batch, 1, seq_len)
        
        # Mask padding positions
        padding_mask = (mask == 0).unsqueeze(1)  # (batch, 1, seq_len)
        scores = scores.masked_fill(padding_mask, float("-inf"))
        
        # Normalize attention weights
        weights = torch.softmax(scores, dim=-1)  # (batch, 1, seq_len)
        weights = torch.nan_to_num(weights)
        
        # Apply attention to values
        pooled = torch.matmul(weights, value)  # (batch, 1, hidden_dim)
        return pooled.squeeze(1)  # (batch, hidden_dim)


class Decoder(nn.Module):
    """LSTM decoder that reconstructs the input sequence from a context sequence."""

    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int, dropout: float) -> None:
        """Create the decoder LSTM and output projection."""
        super().__init__()
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.project = nn.Linear(hidden_dim, output_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input sequence from the context sequence."""
        output, _ = self.lstm(context)
        return self.project(output)


class LSTMAutoencoder(nn.Module):
    """Hybrid LSTM-Autoencoder with attention for per-receiver transaction windows.

    Trained on normal-only windows; fraud is expected to reconstruct poorly
    because it deviates from the normal receiver-history pattern.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        attention: bool = True,
        bottleneck: bool = False,
        attention_pooling: bool = False,
        dropout: float = 0.1,
    ) -> None:
        """Create the encoder, decoder, and the context strategy between them.

        `attention_pooling=True` uses attention to pool encoder outputs to a single
        bottleneck vector, then repeats it across the window.
        
        `bottleneck=True` takes priority over `attention` and `attention_pooling`:
        the decoder is fed only the encoder's final hidden state, repeated across
        the window -- a real information bottleneck. Both `attention=True` and
        `attention=False` instead give the decoder a per-step context it can copy
        from (self-attend or raw per-step outputs), so neither is a genuine
        bottleneck on its own.
        """
        super().__init__()
        self.attention_enabled = attention and not bottleneck and not attention_pooling
        self.bottleneck_enabled = bottleneck
        self.attention_pooling_enabled = attention_pooling and not bottleneck
        self.encoder = Encoder(input_dim, hidden_dim, num_layers, dropout)
        self.attention = SelfAttention(hidden_dim) if self.attention_enabled else None
        self.attention_pooling = AttentionPooling(hidden_dim) if self.attention_pooling_enabled else None
        self.decoder = Decoder(hidden_dim, input_dim, num_layers, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return the reconstructed sequence, same shape as `x`."""
        encoded, (hidden, _) = self.encoder(x)
        if self.bottleneck_enabled:
            context = self._repeat_last_hidden(hidden, window=x.size(1))
        elif self.attention_pooling_enabled:
            pooled = self.attention_pooling(encoded, mask)  # (batch, hidden_dim)
            context = pooled.unsqueeze(1).repeat(1, x.size(1), 1)  # (batch, window, hidden_dim)
        elif self.attention_enabled:
            context = self.attention(encoded, mask)
        else:
            context = encoded
        return self.decoder(context)

    @staticmethod
    def _repeat_last_hidden(hidden: torch.Tensor, window: int) -> torch.Tensor:
        """Repeat the encoder's final-layer hidden state across all `window`
        decode steps. This is the single-vector bottleneck a classic
        seq2seq autoencoder uses, so the decoder cannot copy per-step input."""
        return hidden[-1].unsqueeze(1).repeat(1, window, 1)

    @staticmethod
    def reconstruction_error(
        x: torch.Tensor, reconstructed: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Per-example mean squared error over real (non-padded) steps only."""
        squared_error = (x - reconstructed) ** 2
        squared_error = squared_error.mean(dim=-1) * mask
        return squared_error.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)