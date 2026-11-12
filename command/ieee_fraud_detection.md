### 1. Download the Dataset

Download [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data) from Kaggle and place it under `data/raw/` with a domain-specific folder:

```
data/raw/ieee_fraud_detection/ieee_fraud_detection.csv
data/raw/ieee_fraud_detection/ieee_fraud_detection.zip
```

The loader reads the CSV first and falls back to the zip when no CSV exists. Each dataset should be organized in its own folder under `data/raw/` to support multiple datasets from different domains.

Dataset paths, column names, thresholds, and split fractions are defined in `src/lstm_autoencoder/config.py`.

### 2. Inspect the Dataset

Run the comprehensive dataset inspection pipeline:

```Shell
uv run python -m lstm_autoencoder inspect-data --dataset ieee_fraud_detection
```

The exploratory (IEEEFraudDetectionDatasetExploratory) performs the following operations:

- **Entity cardinality**: Number of unique accounts (composite uid from card1-card6) and average transactions per account
- **History distribution**: Statistics on transactions per account (min, max, median, percentiles)
- **Fraud distribution**: Total fraud transactions, accounts with fraud, and fraud by account history length
- **Fraud position**: Whether fraud appears as first, second, third, or subsequent transaction
- **History before fraud**: Number of normal transactions before fraud occurrences
- **Sequence length availability**: Accounts and sequences available for different window lengths (1, 3, 5, 10, 15, 20)
- **Temporal distribution**: Fraud rate distribution across time periods and temporal drift analysis

The command saves the following artifacts under `artifacts/`:

- `dataset_inspection.txt` - Human-readable inspection report
- `dataset_inspection.json` - Machine-readable inspection results

Optional arguments:

- `--project-root` : Project root that contains `data/raw/`. Defaults to the current working directory.
- `--dataset` : Dataset folder name in `data/raw/` (e.g., `ieee_fraud_detection`). Required.
- `--nrows` : Read only the first N rows of the file. Defaults to all rows. Use it for a quick trial run only, because the file is not time-ordered and a partial read distorts the per-account sequence statistics.

### 3. Probe Signal Learnability

Check whether the fraud label can be learned from the available features:

```Shell
uv run python -m lstm_autoencoder probe-signal --dataset ieee_fraud_detection
```

The signal probe (IEEESignalProbe) trains a strong supervised model (HistGradientBoostingClassifier) and compares it with a shuffled-label control to determine if the target is learnable.

This step is critical: if the best supervised model cannot beat chance, no model (including LSTM-AE) can detect fraud in this dataset.

The command saves the following artifacts under `artifacts/`:

- `signal_report.txt` - Human-readable signal probe report
- `signal_report.json` - Machine-readable signal probe results

Optional arguments:

- `--project-root` : Project root that contains `data/raw/`. Defaults to the current working directory.
- `--dataset` : Dataset folder name in `data/raw/` (e.g., `ieee_fraud_detection`). Required.
- `--nrows` : Read only the first N rows of the file. Defaults to all rows.
- `--max-train-rows` : Maximum number of training rows used by the probe model. Defaults to 5000.
