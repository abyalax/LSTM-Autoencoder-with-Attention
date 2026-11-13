"""The BankSim pipeline must feed the AE the BankSim features, not the PaySim defaults."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from banksim_fixtures import write_synthetic_csv

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.model.config import ModelConfig, SequenceConfig


def test_pipeline_resolves_banksim_features_and_input_dim(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from lstm_autoencoder.pipeline.banksim import banksim_pipeline as module

    write_synthetic_csv(tmp_path, n_customers=120, seed=5)
    config = DatasetConfig(project_root=tmp_path, dataset_name="banksim")
    pipeline = module.BankSimModelTrainingPipeline(
        config,
        sequence_config=SequenceConfig(window_length=8),
        model_config=ModelConfig(bottleneck=True, attention=False, bottleneck_dim=16),
    )
    assert pipeline.model_config.input_dim == 5  # PaySim default until the data is loaded

    captured: dict[str, object] = {}
    original = module.SequenceBuilder.build

    def spy(self, scaled):  # type: ignore[no-untyped-def]
        batch = original(self, scaled)
        captured["batch"], captured["scaled"] = batch, scaled
        return batch

    module.SequenceBuilder.build = spy
    try:
        pipeline._prepare_data()
    finally:
        module.SequenceBuilder.build = original

    columns = pipeline.sequence_config.feature_columns
    assert pipeline.model_config.input_dim == len(columns) == 21
    batch, scaled = captured["batch"], captured["scaled"]
    assert batch.sequences.shape[1:] == (8, 21) and np.isfinite(batch.sequences).all()

    indicators = [c for c in columns if c.startswith("cat_")] + ["is_new_merchant"]
    assert set(np.unique(scaled[indicators].to_numpy())) <= {0.0, 1.0}  # not standardized
