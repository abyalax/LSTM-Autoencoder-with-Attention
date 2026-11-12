# LSTM Autoencoder with Attention for Real-Time Fraud Detection

Research code for an undergraduate thesis on real-time anomaly detection in financial transactions using a hybrid LSTM-Autoencoder with an attention mechanism.

The repository currently provides the dataset validation tooling that precedes modelling:

- Dataset inspection (structure, target balance, sequence viability, leakage candidates)
- A signal probe that checks whether the fraud label can be learned from the available features at all

## Installation

Install [uv](https://docs.astral.sh/uv/), the Python package and project manager used by this repository.

Install the core and development dependencies:

```Shell
uv sync
```

The Python version is pinned in `.python-version`.

## Development

Run the test suite and the code-style checks:

```Shell
uv run pytest
uv run ruff check .
uv run ruff format .
```

# Data layout

Raw files are resolved through `lstm_autoencoder.data.dataset_files.DATASET_FILES`.
Each dataset keeps its untouched source under `data/raw/<dataset>/`. Dataset
processing code can use `DatasetConfig` path accessors for the corresponding
working areas:

```text
data/
├── raw/<dataset>/
├── preprocessed/<dataset>/
├── splits/{train,test,eval}/<dataset>/
└── evaluations/<dataset>/
```

`DatasetConfig.preprocessed_dir`, `split_dir("train" | "test" | "eval")`, and
`evaluations_dir` return these paths. They are paths only; callers create the
directory when writing an artifact.

## Usage
See on command/

_Last Updated 4 November 2026_

