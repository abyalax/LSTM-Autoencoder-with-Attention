"""Per-feature standardization fit on normal-only training rows."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Standardize sequence feature columns, fit only on normal-only training rows.

    Unnormalized `amount` (can run into the millions) otherwise dominates the
    reconstruction MSE and drowns out the gradient from the other, much smaller,
    features. Fitting on train-normal only (never val/test, never fraud rows)
    keeps the scale consistent with how the autoencoder itself is trained.
    """

    def __init__(self, feature_columns: tuple[str, ...]) -> None:
        """Bind the scaler to the feature columns it will standardize."""
        self._feature_columns = list(feature_columns)
        self._scaler = StandardScaler()

    def fit(self, built: pd.DataFrame, fit_mask: pd.Series) -> FeatureScaler:
        """Fit mean/std from the rows selected by `fit_mask`."""
        self._scaler.fit(built.loc[fit_mask, self._feature_columns])
        return self

    def transform(self, built: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `built` with feature columns standardized."""
        scaled = built.copy()
        scaled[self._feature_columns] = self._scaler.transform(built[self._feature_columns])
        return scaled