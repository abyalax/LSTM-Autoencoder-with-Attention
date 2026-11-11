"""Padding must not change what the LSTM-AE computes for the real steps (needs PyTorch).

The encoder packs each right-padded window by its real length, so adding padded columns
(or putting garbage in them) changes neither the encoder state nor the reconstruction/scores
of the real steps. This is the regression test for the old left-padded, unpacked encoder,
whose state depended on how many zero steps preceded the real ones.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lstm_autoencoder.model.lstm_autoencoder import Encoder, LSTMAutoencoder  # noqa: E402

N_FEATURES = 5
REAL = 3


def _batch(window: int, garbage: float = 0.0, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Two examples with REAL real steps (identical across windows), right-padded to `window`."""
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(2, REAL, N_FEATURES, generator=generator)
    x = torch.zeros(2, window, N_FEATURES)
    x[:, :REAL] = real
    if window > REAL:
        x[:, REAL:] = garbage * torch.randn(2, window - REAL, N_FEATURES, generator=generator)
    mask = torch.zeros(2, window)
    mask[:, :REAL] = 1.0
    return x, mask


def test_encoder_state_ignores_padding_length_and_padding_content():
    torch.manual_seed(1)
    encoder = Encoder(N_FEATURES, 8, num_layers=2, dropout=0.0).eval()
    x_short, m_short = _batch(REAL)
    x_long, m_long = _batch(10, garbage=0.0)
    x_noisy, m_noisy = _batch(10, garbage=5.0)

    out_short, (h_short, c_short) = encoder(x_short, m_short)
    for x, m in ((x_long, m_long), (x_noisy, m_noisy)):
        out, (h, c) = encoder(x, m)
        torch.testing.assert_close(h, h_short)
        torch.testing.assert_close(c, c_short)
        torch.testing.assert_close(out[:, :REAL], out_short)
        assert torch.all(out[:, REAL:] == 0)  # padded positions carry no state


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attention": False},  # plain
        {"attention": True},  # diagonal-masked self-attention
        {"attention": False, "bottleneck": True},
        {"attention": False, "attention_pooling": True},
        {"attention": False, "attention_pooling": True, "bottleneck_dim": 4},
    ],
)
def test_reconstruction_of_real_steps_does_not_depend_on_padding(kwargs):
    torch.manual_seed(2)
    model = LSTMAutoencoder(N_FEATURES, 8, num_layers=2, dropout=0.0, **kwargs).eval()
    x_short, m_short = _batch(REAL)
    x_long, m_long = _batch(10, garbage=5.0)
    with torch.no_grad():
        rec_short = model(x_short, m_short)
        rec_long = model(x_long, m_long)
        torch.testing.assert_close(rec_long[:, :REAL], rec_short, atol=1e-5, rtol=1e-5)
        for strategy in ("mean", "last"):
            a = LSTMAutoencoder.reconstruction_error(x_short, rec_short, m_short, strategy)
            b = LSTMAutoencoder.reconstruction_error(x_long, rec_long, m_long, strategy)
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_last_error_is_the_current_transaction_and_mean_averages_real_steps_only():
    x = torch.zeros(2, 4, 2)
    rec = torch.zeros(2, 4, 2)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    rec[0, 0] = 1.0  # error 1 at step 0, 0 at steps 1-2
    rec[0, 2] = 2.0  # error 4 at the LAST real step
    rec[1, 0] = 3.0  # single-step window: mean == last
    rec[0, 3] = 99.0  # padded step: must be ignored by both strategies
    last = LSTMAutoencoder.reconstruction_error(x, rec, mask, "last")
    mean = LSTMAutoencoder.reconstruction_error(x, rec, mask, "mean")
    torch.testing.assert_close(last, torch.tensor([4.0, 9.0]))
    torch.testing.assert_close(mean, torch.tensor([5.0 / 3.0, 9.0]))
    with pytest.raises(ValueError):
        LSTMAutoencoder.reconstruction_error(x, rec, mask, "max")


def test_higher_reconstruction_error_ranks_fraud_above_normal():
    from lstm_autoencoder.evaluation.metrics import RankingMetrics

    errors = np.array([0.1, 0.2, 0.15, 5.0, 4.0])
    labels = np.array([0, 0, 0, 1, 1])
    assert RankingMetrics.roc_auc(errors, labels) == pytest.approx(1.0)
