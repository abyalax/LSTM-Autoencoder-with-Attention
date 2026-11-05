### 1. Download the Dataset

Download [Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1/data) from Kaggle and place it under `data/raw/` with a domain-specific folder:

```
data/raw/paysim_synthetic_financial_datasets/paysim_synthetic_financial.csv
data/raw/paysim_synthetic_financial_datasets/paysim_synthetic_financial.zip
```

The loader reads the CSV first and falls back to the zip when no CSV exists. Each dataset should be organized in its own folder under `data/raw/` to support multiple datasets from different domains.

Dataset paths, column names, thresholds, and split fractions are defined in `src/lstm_autoencoder/config.py`.

### 2. Inspect the Dataset

Run the full set of dataset checks:

```Shell
uv run python -m lstm_autoencoder inspect-data --dataset paysim_synthetic_financial
```

The explorer (PaySimSyntheticFinancialDatasetsExplorer) performs the following operations:

- **Structure inspection**: column count, data types, missing values, duplicate rows, and unique value counts
- **Target inspection**: fraud prevalence overall and by transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
- **isFlaggedFraud inspection**: checks overlap between isFlaggedFraud and isFraud columns (potential artifact)
- **Account viability inspection**: evaluates sender account history lengths and calculates percentage of accounts supporting each window length (5, 10, 20 transactions)
- **Chronological split simulation**: 70/15/15 split ordered by step column, reports fraud rate per split
- **Univariate signal probing**: computes Mann-Whitney U based AUC for each numeric feature, both with and without leakage columns (oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest)

The loader automatically applies explicit string dtypes for `nameOrig`, `nameDest`, and `type` columns to prevent incorrect type inference on large datasets (6.3M rows) during groupby operations.

The command saves the following artifacts under `artifacts/`:

- `dataset_report.txt`
- `dataset_report.json`

Optional arguments:

- `--project-root` : Project root that contains `data/raw/`. Defaults to the current working directory.
- `--dataset` : Dataset folder name in `data/raw/` (e.g., `paysim_synthetic_financial`). Required.
- `--nrows` : Read only the first N rows of the file. Defaults to all rows. Use it for a quick trial run only, because the file is not time-ordered and a partial read distorts the per-account sequence statistics.

### 3. Probe the Label Signal

Train a strong supervised model on a chronological split and compare it with a shuffled-label control. If the best supervised model cannot beat chance, no model can detect fraud in this dataset.

```Shell
uv run python -m lstm_autoencoder probe-signal --dataset paysim_synthetic_financial --max-train-rows 1000000
```

The probe saves the following artifact under `artifacts/`:

- `signal_report.txt`

The report ends with one of three verdicts:

- No signal: test AUC below 0.51 and average precision below 1.15 times the prevalence
- Weak signal: test AUC below 0.60
- Learnable: test AUC of 0.60 or higher

Optional arguments:

- `--project-root` : Project root that contains `data/raw/`. Defaults to the current working directory.
- `--dataset` : Dataset folder name in `data/raw/` (e.g., `paysim_synthetic_financial`). Required.
- `--nrows` : Read only the first N rows of the file. Defaults to all rows. Same caveat as `inspect-data`.
- `--max-train-rows` : Maximum number of training rows sampled for the probe model. Defaults to 1,500,000. Lower it to speed up the run. Validation and test rows are never subsampled.

### 4. Run the Ablation Baselines

Train three logistic-regression variants on the receiver-stratified split — the full feature set, the full set minus the causal history features, and the history features alone — and compare them overall and per receiver history-length bucket (`1`, `2-4`, `5-9`, `>=10`).

```Shell
uv run python -m lstm_autoencoder ablation --dataset paysim_synthetic_financial
```

The evaluator (BaselineEvaluator) performs the following operations:

- **Feature encoding**: one-hot encodes `type`, drops leakage columns (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) and `isFlaggedFraud`
- **Receiver-stratified split**: 70/15/15 split by `nameDest`, stratified by per-receiver fraud rate, same scheme used by `probe-signal`
- **Variant training**: fits a standardized `LogisticRegression` per variant (`full`, `no_history`, `history_only`) on train
- **Stratified scoring**: reports AUC/AP overall (val, test) and per history bucket on test, for every variant

The command saves the following artifacts under `artifacts/`:

- `ablation_report.txt`
- `ablation_report.json`

Optional arguments:

- `--project-root` : Project root that contains `data/raw/`. Defaults to the current working directory.
- `--dataset` : Dataset folder name in `data/raw/` (e.g., `paysim_synthetic_financial`). Required.
- `--nrows` : Read only the first N rows of the file. Defaults to all rows. Same caveat as `inspect-data`.

### 5. Train the LSTM-Autoencoder

Build fixed-length per-receiver transaction windows, train the hybrid LSTM-Autoencoder with attention on normal-only windows, and evaluate reconstruction error as an anomaly score on the held-out test set.

```Shell
uv run python -m lstm_autoencoder train-model --dataset paysim_synthetic_financial --window 5
```

The pipeline (PaySimModelTrainingPipeline) performs the following operations:

- **Sequence building**: left-pads each receiver's last `--window` transactions into fixed-length windows (`amount`, `step_diff`, `log_amount_ratio`, `history_length`, `is_new_receiver`)
- **Receiver-stratified split**: 70/15/15 split by `nameDest`, stratified by per-receiver fraud rate; the training set is further restricted to normal (non-fraud) windows only
- **Training**: fits the LSTM-Autoencoder with attention via Adam, with early stopping on validation reconstruction-error AUC
- **Evaluation**: scores the held-out test set (fraud included) by reconstruction-error AUC and average precision

The model architecture supports three variants via the `--bottleneck` flag:
- **Default (attention)**: Decoder receives self-attended context over encoder outputs (attends to other real steps, never to itself unless alone)
- **Bottleneck (`--bottleneck`)**: Decoder receives only the encoder's final hidden state, repeated across the window — a true information bottleneck forcing the model to compress sequence information
- **Plain**: Decoder receives raw per-step encoder outputs (can copy directly)

The command saves the following artifacts under `artifacts/`:

- `training_report.txt` or `multi_seed_training_report.txt` (when `--seeds` is used)
- `training_report.json` or `multi_seed_training_report.json` (when `--seeds` is used)
- `lstm_ae_w{window}_{attn|bottleneck|plain}_seed{seed}_best.pt` (best checkpoint, under `artifacts/model/paysim_synthetic_financial/`)

Optional arguments:

- `--project-root` : Project root that contains `data/raw/`. Defaults to the current working directory.
- `--dataset` : Dataset folder name in `data/raw/` (e.g., `paysim_synthetic_financial`). Required.
- `--nrows` : Read only the first N rows of the file. Defaults to all rows. Same caveat as `inspect-data`.
- `--window` : Window length for per-receiver sequences. Defaults to 5. Matches the grid explored in `inspect-data` (1, 3, 5, 10).
- `--seeds` : Comma-separated random seeds for repeated experiments (e.g., `42,123,2026`). When set, trains once per seed on the same data split and reports mean/std across runs to separate genuine improvement from training noise. Default is single run with seed 42.
- `--bottleneck` : Use a single-vector bottleneck decoder (final encoder hidden state, repeated across the window) instead of attention or raw per-step context. Default is False (uses attention).

### 6. Probe hybrid signal

Correlate a trained LSTM-Autoencoder checkpoint's reconstruction error with the logistic regression baseline's score on the same test rows, before committing to a full hybrid retrain. Low correlation suggests the reconstruction error could add complementary signal to the LR; high correlation suggests it is mostly redundant with what the LR already captures from the same underlying features.

```Shell
uv run python -m lstm_autoencoder probe-hybrid-signal --dataset paysim_synthetic_financial --window 5 --seed 42
```

The probe (HybridSignalProbe) performs the following operations:

- **Autoencoder scoring**: Loads the trained checkpoint (no retraining) and scores the test windows by reconstruction error
- **Baseline scoring**: Fits the LR 'full' baseline fresh and scores the same test rows
- **Alignment check**: Ensures both pipelines score the SAME test rows (same dataset, nrows, and receiver split seed)
- **Correlation analysis**: Computes Pearson and Spearman correlation between the two signals
- **Error stratification**: Reports mean AE reconstruction error split by label and, for fraud rows, by whether LR scored it low (LR's likely blind spot) or high (LR already confident)

The probe saves the following artifacts under `artifacts/`:

- `hybrid_signal_probe_report.txt`
- `hybrid_signal_probe_report.json`

Optional arguments:

- `--project-root` : Project root that contains `data/raw/`. Defaults to the current working directory.
- `--dataset` : Dataset folder name in `data/raw/` (e.g., `paysim_synthetic_financial`). Required.
- `--nrows` : Read only the first N rows of the file. Defaults to all rows. Same caveat as `inspect-data`.
- `--window` : Window length for per-receiver sequences. Defaults to 5. Matches the grid explored in `inspect-data` (1, 3, 5, 10).
- `--seed` : Seed used when the checkpoint was trained (for its filename). Default is 42.
- `--bottleneck` : Match the checkpoint's architecture: bottleneck decoder instead of attention. Default is False.
- `--checkpoint` : Path to the trained checkpoint. Defaults to the standard train-model naming convention (`lstm_ae_w{window}_{attn|bottleneck|plain}_seed{seed}_best.pt`).

_Last Updated 5 November 2026_