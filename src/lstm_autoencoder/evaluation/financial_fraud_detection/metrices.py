"""Ranking metrics for imbalanced binary classification.

This module re-exports the common RankingMetrics from the base evaluation module
for backward compatibility.
"""

from __future__ import annotations

from lstm_autoencoder.evaluation.metrics import RankingMetrics

__all__ = ["RankingMetrics"]
