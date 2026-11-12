# Operations for New Datasets

This guide covers the operational commands to run when adding a new dataset to the LSTM-Autoencoder project.

## 1. Prepare the Dataset

Place your dataset file(s) under `data/raw/{dataset_name}/`:

```
data/raw/{dataset_name}/{dataset_name}.csv
data/raw/{dataset_name}/{dataset_name}.zip  # optional, as fallback
```

The loader reads the CSV first and falls back to the zip when no CSV exists.

## 2. Create a Schema

Create a schema file in `src/lstm_autoencoder/data/{dataset_name}/schema.py`:

```python
from dataclasses import dataclass
from typing import ClassVar

from lstm_autoencoder.data.base_schema import BaseSchema


@dataclass(frozen=True)
class YourDatasetSchema(BaseSchema):
    """Schema definition for your dataset."""

    # Required fields
    target_column: ClassVar[str] = "is_fraud"  # Your fraud label column
    timestamp_column: ClassVar[str] = "timestamp"  # Your timestamp column
    sender_column: ClassVar[str] = "account_id"  # Your sender/account column
    receiver_column: ClassVar[str] = "receiver_id"  # Your receiver column (optional)

    # Optional: column dtypes to prevent type inference issues on large datasets
    dtypes: ClassVar[dict[str, str]] = {
        "account_id": "string",
        "receiver_id": "string",
        # Add other string columns as needed
    }

    # Optional: column to drop from analysis
    columns_to_drop: ClassVar[tuple[str, ...]] = (
        "unnecessary_column",
        # Add columns to exclude
    )
```

Reference existing schemas:
- `src/lstm_autoencoder/data/banksim/schema.py`
- `src/lstm_autoencoder/data/ieee_fraud_detection/schema.py`
- `src/lstm_autoencoder/data/paysim_synthetic_financial/schema.py`

## 3. Register Dataset in CLI

Add your dataset to the schema mapping in `src/lstm_autoencoder/cli/runner.py`:

```python
from lstm_autoencoder.data.your_dataset.schema import YourDatasetSchema

# In run_inspect_all_datasets method:
schemas = {
    "banksim": BankSimSchema,
    "financial_fraud_detection": FinancialFraudDetectionSchema,
    "ieee_fraud_detection": IEEEFraudDetectionSchema,
    "paysim_synthetic_financial": PaySimSchema,
    "your_dataset": YourDatasetSchema,  # Add your dataset
}
```

## 4. Inspect the Dataset

Run the unified dataset inspection:

```Shell
uv run python -m lstm_autoencoder inspect-data --dataset your_dataset
```

The inspector performs the following operations:

- **Entity cardinality**: Number of unique accounts, total transactions, average transactions per account
- **History distribution**: Distribution of transaction counts per account
- **Fraud distribution**: How many frauds occur on accounts with history
- **Fraud position**: Does fraud appear more often as first or subsequent transaction
- **History before fraud**: How many normal transactions before fraud
- **Sequence length availability**: Are there enough sequences with length 3, 5, 10, etc.
- **Temporal distribution**: Is fraud distributed throughout the dataset period

The command saves artifacts under `artifacts/report/your_dataset/inspector/YourDatasetExploratory/{timestamp}/`:

- `dataset_inspection.txt`
- `dataset_inspection.json`

To create a custom exploratory for your dataset:

1. Create a new module under `src/lstm_autoencoder/data/your_dataset/`
2. Create a `dataset_exploratory.py` file with a class that extends `BaseDatasetExploratory`
3. Implement the `get_schema()` method to return your dataset's schema
4. Add your exploratory to the mapping in `src/lstm_autoencoder/cli/runner.py`

Example:
```python
from lstm_autoencoder.data.base_exploratory import BaseDatasetExploratory
from lstm_autoencoder.data.your_dataset.schema import YourDatasetSchema

class YourDatasetExploratory(BaseDatasetExploratory):
    def get_schema(self) -> YourDatasetSchema:
        return YourDatasetSchema()
```

Optional arguments:

- `--project-root`: Project root that contains `data/raw/`. Defaults to current working directory.
- `--dataset`: Dataset folder name in `data/raw/` (e.g., `your_dataset`). Required.
- `--nrows`: Read only the first N rows of the file. Defaults to all rows. Use for quick trial runs only.

## 5. Probe the Label Signal (Optional)

Train a strong supervised model on a chronological split and compare it with a shuffled-label control. This checks if the dataset has learnable fraud patterns.

```Shell
uv run python -m lstm_autoencoder probe-signal --dataset your_dataset --max-train-rows 1000000
```

The probe saves the following artifact under `artifacts/report/your_dataset/signal_probe/SignalProbe/{timestamp}/`:

- `signal_probe_report.txt`
- `signal_probe_report.json`

The report ends with one of three verdicts:

- No signal: test AUC below 0.51 and average precision below 1.15 times the prevalence
- Weak signal: test AUC below 0.60
- Learnable: test AUC of 0.60 or higher

Optional arguments:

- `--project-root`: Project root that contains `data/raw/`. Defaults to current working directory.
- `--dataset`: Dataset folder name in `data/raw/` (e.g., `your_dataset`). Required.
- `--nrows`: Read only the first N rows of the file. Defaults to all rows.
- `--max-train-rows`: Maximum number of training rows sampled for the probe model. Defaults to 1,500,000. Lower it to speed up the run. Validation and test rows are never subsampled.

## 6. Run Training

Once the dataset is registered and inspected, train models using the standard pipeline:

```Shell
uv run python -m lstm_autoencoder train --dataset your_dataset --window 3 --seeds 42
```

For multiple seeds:

```Shell
uv run python -m lstm_autoencoder train --dataset your_dataset --window 3 --seeds 42 123 2026
```

For multiple windows:

```Shell
uv run python -m lstm_autoencoder train --dataset your_dataset --window 1 3 5 10 --seeds 42
```

With attention mechanism:

```Shell
uv run python -m lstm_autoencoder train --dataset your_dataset --window 3 --seeds 42 --attention
```

The command saves model checkpoints and training reports under `artifacts/report/your_dataset/model_training/ModelTrainingPipeline/{timestamp}/`.

Optional arguments:

- `--project-root`: Project root that contains `data/raw/`. Defaults to current working directory.
- `--dataset`: Dataset folder name in `data/raw/` (e.g., `your_dataset`). Required.
- `--window`: Sequence window length(s). Can be single value or multiple (e.g., `1 3 5 10`). Required.
- `--seeds`: Random seed(s) for reproducibility. Can be single value or multiple (e.g., `42 123 2026`). Defaults to 42.
- `--attention`: Enable attention mechanism.
- `--hidden-dim`: Hidden dimension for LSTM. Defaults to 64.
- `--bottleneck-dim`: Bottleneck dimension for autoencoder. Defaults to 16.
- `--epochs`: Maximum number of training epochs. Defaults to 50.
- `--batch-size`: Batch size for training. Defaults to 256.
- `--learning-rate`: Learning rate for optimizer. Defaults to 0.001.
- `--early-stopping-patience`: Patience for early stopping. Defaults to 10.

## 7. Evaluate Standalone Model

Evaluate the trained LSTM-Autoencoder as a standalone model:

```Shell
uv run python -m lstm_autoencoder evaluate-standalone --dataset your_dataset --window 3 --seeds 42
```

For multiple seeds:

```Shell
uv run python -m lstm_autoencoder evaluate-standalone --dataset your_dataset --window 3 --seeds 42 123 2026
```

For multiple windows:

```Shell
uv run python -m lstm_autoencoder evaluate-standalone --dataset your_dataset --window 1 3 5 10 --seeds 42
```

The command saves evaluation reports under `artifacts/report/your_dataset/standalone_evaluator/StandaloneEvaluator/{timestamp}/`.

Optional arguments:

- `--project-root`: Project root that contains `data/raw/`. Defaults to current working directory.
- `--dataset`: Dataset folder name in `data/raw/` (e.g., `your_dataset`). Required.
- `--window`: Sequence window length(s). Can be single value or multiple. Required.
- `--seeds`: Random seed(s) for reproducibility. Can be single value or multiple. Defaults to 42.
- `--attention`: Enable attention mechanism evaluation.
- `--hidden-dim`: Hidden dimension for LSTM. Defaults to 64.
- `--bottleneck-dim`: Bottleneck dimension for autoencoder. Defaults to 16.
- `--split`: Split to evaluate on (`val` or `test`). Defaults to `test`.

## 8. Run Baseline Evaluation

Evaluate baseline models (Logistic Regression, GBM, unsupervised baselines):

```Shell
uv run python -m lstm_autoencoder evaluate-baseline --dataset your_dataset --seeds 42
```

The command saves baseline reports under `artifacts/report/your_dataset/baseline/BaselineEvaluator/{timestamp}/`.

Optional arguments:

- `--project-root`: Project root that contains `data/raw/`. Defaults to current working directory.
- `--dataset`: Dataset folder name in `data/raw/` (e.g., `your_dataset`). Required.
- `--seeds`: Random seed(s) for reproducibility. Can be single value or multiple. Defaults to 42.
- `--split`: Split to evaluate on (`val` or `test`). Defaults to `test`.

## 9. Hybrid Classifier (Optional)

Combine LSTM-Autoencoder with baseline models:

```Shell
uv run python -m lstm_autoencoder hybrid --dataset your_dataset --window 3 --seeds 42
```

The command saves hybrid classifier reports under `artifacts/report/your_dataset/hybrid_classifier/HybridClassifier/{timestamp}/`.

Optional arguments:

- `--project-root`: Project root that contains `data/raw/`. Defaults to current working directory.
- `--dataset`: Dataset folder name in `data/raw/` (e.g., `your_dataset`). Required.
- `--window`: Sequence window length. Required.
- `--seeds`: Random seed(s) for reproducibility. Can be single value or multiple. Defaults to 42.
- `--attention`: Enable attention mechanism.
- `--hidden-dim`: Hidden dimension for LSTM. Defaults to 64.
- `--bottleneck-dim`: Bottleneck dimension for autoencoder. Defaults to 16.
- `--split`: Split to evaluate on (`val` or `test`). Defaults to `test`.

## 10. Summarize Results

Generate a summary report from all experiments:

```Shell
uv run python -m lstm_autoencoder summarize-results --dataset your_dataset
```

The command saves a markdown summary under `artifacts/summary/results_summary_{dataset_name}.md`.

Optional arguments:

- `--project-root`: Project root that contains `data/raw/`. Defaults to current working directory.
- `--dataset`: Dataset folder name in `data/raw/` (e.g., `your_dataset`). Required.
- `--out`: Output path for the summary markdown. Defaults to `artifacts/summary/results_summary_{dataset_name}.md`.

## 11. Diagnose Scores (Optional)

Diagnose reconstruction score behavior:

```Shell
uv run python -m lstm_autoencoder diagnose-scores --dataset your_dataset --window 3 --seeds 42
```

Saves `score_diagnostics_report.txt` and `.json` under `artifacts/report/your_dataset/score_diagnostics/ScoreDiagnostics/{timestamp}/`.

Optional arguments: `--attention`, `--hidden-dim`, `--bottleneck-dim`, `--seeds` (default 42), `--split` (`val` or `test`, default `val`), `--window`, `--nrows`.

## 12. Probe Hybrid Signal (Optional)

Probe correlation between LSTM-AE reconstruction error and baseline scores:

```Shell
uv run python -m lstm_autoencoder probe-hybrid-signal --dataset your_dataset --window 3 --seeds 42
```

Saves `hybrid_signal_probe_report.txt` and `.json` under `artifacts/report/your_dataset/hybrid_signal_probe/HybridSignalProbe/{timestamp}/`.

Optional arguments: `--attention`, `--hidden-dim`, `--bottleneck-dim`, `--seeds` (default 42), `--split` (`val` or `test`, default `val`), `--window`, `--checkpoint-path`.

## Complete Workflow Example

```Shell
# 1. Inspect dataset
uv run python -m lstm_autoencoder inspect-data --dataset your_dataset

# 2. Probe signal strength
uv run python -m lstm_autoencoder probe-signal --dataset your_dataset --max-train-rows 1000000

# 3. Train models (multiple windows and seeds)
uv run python -m lstm_autoencoder train --dataset your_dataset --window 1 3 5 10 --seeds 42 123 2026

# 4. Evaluate standalone
uv run python -m lstm_autoencoder evaluate-standalone --dataset your_dataset --window 1 3 5 10 --seeds 42 123 2026

# 5. Evaluate baselines
uv run python -m lstm_autoencoder evaluate-baseline --dataset your_dataset --seeds 42 123 2026

# 6. Hybrid classifier
uv run python -m lstm_autoencoder hybrid --dataset your_dataset --window 3 --seeds 42 123 2026

# 7. Summarize results
uv run python -m lstm_autoencoder summarize-results --dataset your_dataset
```

## Directory Structure

```
lstm-autoencoder/
├── data/
│   └── raw/
│       └── your_dataset/
│           ├── your_dataset.csv
│           └── your_dataset.zip
├── src/
│   └── lstm_autoencoder/
│       ├── data/
│       │   └── your_dataset/
│       │       ├── schema.py
│       │       ├── dataset_exploratory.py
│       │       └── exploratory.py (optional)
│       ├── pipeline/
│       │   └── your_dataset/ (optional)
│       │       └── your_dataset_pipeline.py
│       └── evaluation/
│           └── your_dataset/ (optional)
├── artifacts/
│   ├── report/
│   │   └── your_dataset/
│   │       ├── inspector/
│   │       ├── signal_probe/
│   │       ├── model_training/
│   │       ├── standalone_evaluator/
│   │       ├── baseline/
│   │       ├── hybrid_classifier/
│   │       ├── score_diagnostics/
│   │       └── hybrid_signal_probe/
│   └── summary/
│       └── results_summary_your_dataset.md
└── command/
    └── your_dataset.md (optional, dataset-specific docs)
```

## Common Issues

1. **Type inference errors on large datasets**: Specify explicit dtypes in your schema's `dtypes` field
2. **Missing columns**: Ensure your CSV has all required columns (target, timestamp, sender)
3. **Temporal split issues**: Ensure your timestamp column is parseable by pandas
4. **Report path issues**: The auto-detection uses dataset name from `--dataset` argument
5. **CUDA out of memory**: Reduce `--batch-size` or `--hidden-dim` when training large models
