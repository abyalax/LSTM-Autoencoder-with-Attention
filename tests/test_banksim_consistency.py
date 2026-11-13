"""BankSim workflows must agree on the split and on which learner a variant uses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from banksim_fixtures import write_synthetic_csv

from lstm_autoencoder.config import DatasetConfig
from lstm_autoencoder.data.banksim.causal_features import CausalFeatureBuilder, account_identity
from lstm_autoencoder.data.banksim.schema import BankSimSchema
from lstm_autoencoder.evaluation.banksim.baseline import MODEL_GBM, MODEL_LR, BaselineEvaluator
from lstm_autoencoder.evaluation.banksim.split import split_customers


@pytest.fixture
def dataset_config(tmp_path: Path) -> DatasetConfig:
    write_synthetic_csv(tmp_path, n_customers=160, seed=3)
    return DatasetConfig(project_root=tmp_path, dataset_name="banksim")


def _built(dataset_config: DatasetConfig):
    schema = BankSimSchema()
    from lstm_autoencoder.data.loader import DatasetLoader

    loader = DatasetLoader(dataset_config, schema)
    loader._parse_timestamps = lambda series: series
    loader._normalize_target = lambda series: series
    frame = loader.load().frame
    return CausalFeatureBuilder(dataset_config, schema).build(frame), schema


def test_split_is_customer_disjoint_and_label_free(dataset_config: DatasetConfig) -> None:
    built, schema = _built(dataset_config)
    account = account_identity(built, schema)
    groups = split_customers(account)

    ids = {name: set(index) for name, index in groups.items()}
    assert ids["train"].isdisjoint(ids["val"]) and ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    assert ids["train"] | ids["val"] | ids["test"] == set(account.unique())

    flipped = built.copy()
    flipped[schema.target_column] = 1 - flipped[schema.target_column]
    again = split_customers(account_identity(flipped, schema))
    assert {name: set(index) for name, index in again.items()} == ids


def test_baseline_split_is_the_shared_customer_split(dataset_config: DatasetConfig) -> None:
    built, schema = _built(dataset_config)
    expected = split_customers(account_identity(built, schema))
    data = BaselineEvaluator(dataset_config)._prepare_full_data()
    for name, customers in expected.items():
        rows = np.flatnonzero(account_identity(built, schema).isin(customers).to_numpy())
        assert np.array_equal(np.sort(data.splits[name]), rows)


def test_gbm_variants_are_really_gbm(dataset_config: DatasetConfig) -> None:
    evaluator = BaselineEvaluator(dataset_config)
    data = evaluator._prepare_full_data()
    variants = {variant.name: variant for variant in evaluator._define_variants(data)}

    assert variants["full"].model == MODEL_LR
    assert variants["full_gbm"].model == MODEL_GBM
    assert variants["static_gbm"].model == MODEL_GBM
    assert set(variants["static_gbm"].columns) < set(variants["full_gbm"].columns)

    # Regression test: a GBM variant used to be fit as a logistic regression, so two
    # "different" variants returned bit-identical scores.
    full_lr = evaluator._fit_variant(variants["full"], data)
    full_gbm = evaluator._fit_variant(variants["full_gbm"], data)
    assert not np.allclose(full_lr, full_gbm)


def test_pipeline_uses_the_shared_customer_split(dataset_config: DatasetConfig) -> None:
    pytest.importorskip("torch")
    from lstm_autoencoder.pipeline.banksim.banksim_pipeline import BankSimModelTrainingPipeline

    built, schema = _built(dataset_config)
    pipeline = BankSimModelTrainingPipeline(dataset_config)
    expected = split_customers(account_identity(built, schema))
    got = pipeline.split_receivers(built)
    assert {name: set(index) for name, index in got.items()} == {name: set(index) for name, index in expected.items()}
