"""Per-feature standardization fit on normal-only training rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Log1p heavy-tailed columns, then standardize, fit only on normal-only training rows.

    Unnormalized `amount` (can run into the millions) otherwise dominates the
    reconstruction MSE and drowns out the gradient from the other, much smaller,
    features. StandardScaler alone is not enough: it is a mean/std fit, and a
    few extreme values in a heavy-tailed, non-negative column (amount,
    history_length) inflate the fitted std and compress the signal every other
    row carries in that column -- log1p first makes the column roughly
    symmetric so the subsequent z-score is meaningful. Fitting on train-normal
    only (never val/test, never fraud rows) keeps the scale consistent with
    how the autoencoder itself is trained.
    """

    def __init__(self, feature_columns: tuple[str, ...], log1p_columns: tuple[str, ...] = ()) -> None:
        """Bind the scaler to the feature columns it will standardize.

        `log1p_columns` names the heavy-tailed, non-negative columns (subset of
        `feature_columns`) that get log1p'd before the StandardScaler fit/transform.
        """
        self._feature_columns = list(feature_columns)
        self._log1p_columns = [c for c in log1p_columns if c in feature_columns]
        self._scaler = StandardScaler()

    def _pre_transform(self, built: pd.DataFrame) -> pd.DataFrame:
        """Select feature columns and log1p the configured heavy-tailed ones."""
        subset = built[self._feature_columns].copy()
        if self._log1p_columns:
            subset[self._log1p_columns] = np.log1p(subset[self._log1p_columns])
        return subset

    def fit(self, built: pd.DataFrame, fit_mask: pd.Series) -> FeatureScaler:
        """Fit mean/std from the rows selected by `fit_mask`, after log1p."""
        self._scaler.fit(self._pre_transform(built.loc[fit_mask]))
        return self

    def transform(self, built: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `built` with feature columns log1p'd (where configured) and standardized."""
        scaled = built.copy()
        scaled[self._feature_columns] = self._scaler.transform(self._pre_transform(built))
        return scaled
