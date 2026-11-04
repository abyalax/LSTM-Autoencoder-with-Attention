# IMPORTANT

dataset ini **belum layak dilanjutkan dalam bentuknya sekarang**. Hasil probe-signal yang dijalankan menunjukkan masalah fundamental, bukan hanya "modelnya kurang bagus".

## Bukti dari hasil yang dijalankan

- **Upper bound supervised cuma AUC 0.594** (AP 0.0442, hampir sama dengan baseline acak 0.0361). Ini pakai label asli, fitur lengkap tanpa leakage, model terbaik (XGBoost/LightGBM-tier). Kalau model dengan akses penuh ke label aja cuma segitu, LSTM-AE unsupervised (yang kerjanya cuma belajar pola normal tanpa lihat label) nyaris pasti tidak akan lebih baik — langit-langitnya sudah rendah.
- **Kontrol label-diacak AUC 0.493** — hampir sama dengan hasil label asli (0.594). Gap-nya cuma ~0.1, sangat tipis untuk 5 juta baris data.
- **Semua fitur numerik AUC univariat ≈ 0.50** (amount, velocity_score, geo_anomaly_score, spending_deviation_score — termasuk kolom yang namanya "anomaly/deviation score" yang harusnya didesain untuk menyorot fraud, tapi ternyata tidak berkorelasi sama sekali dengan `is_fraud`).
- **Indeks dispersi fraud per akun = 0.9572** (mendekati 1.0 = acak). Artinya fraud tidak mengelompok di akun/pola tertentu — konsisten dengan label yang di-assign random saat dataset dibuat, bukan hasil simulasi perilaku fraud yang realistis.
- Kesimpulan pada hasil run *"Sinyal sangat lemah, hasil model deep kemungkinan tidak bermakna."*

## Kenapa ini fatal untuk proposal ini spesifik

RQ1–RQ4 dan novelty kamu (G1–G3) semuanya bertumpu pada asumsi bahwa ada pola sekuensial/relasional yang bisa "ditangkap" attention. Kalau sinyalnya memang nyaris tidak ada di level data (bukan di level model), maka:
- Eksperimen ablation attention (E2) tidak akan menunjukkan apa-apa yang bermakna — bobot attention cuma akan menyorot noise.
- Perbandingan baseline vs model usulan (E1) kemungkinan besar semua model akan convergen ke performa mirip-mirip rendah, bukan karena LSTM-AE+attention gagal, tapi karena datanya memang tidak punya pola untuk dipelajari.
- Reviewer SINTA 3 akan mempertanyakan validitas eksperimen begitu melihat AUPRC yang nyaris sama dengan prevalensi di semua model.

If you still want to check and re run, follow this step

### 1. Download the Dataset

Download [Financial Transactions Dataset for Fraud Detection](https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection/data) from Kaggle and place it under `data/raw/` with a domain-specific folder:

```
data/raw/financial_fraud_detection/financial_fraud_detection.csv
data/raw/financial_fraud_detection/financial_fraud_detection.zip
```

The loader reads the CSV first and falls back to the zip when no CSV exists. Each dataset should be organized in its own folder under `data/raw/` to support multiple datasets from different domains.

Dataset paths, column names, thresholds, and split fractions are defined in `src/lstm_autoencoder/config.py`.

### 2. Inspect the Dataset

Run the full set of dataset checks:

```Shell
uv run python -m lstm_autoencoder inspect-data --dataset financial_fraud_detection
```

The report covers seven sections:

- Structure: columns, data types, missing values, duplicates, time range
- Target: fraud rate overall, per category, and per period
- Amount: distribution per class
- Sequences: history length per sender account and window viability
- Signal: univariate AUC of raw and causal features
- Leakage: columns that look derived from the label
- Split: a simulated chronological 70/15/15 split

The command saves the following artifacts under `artifacts/`:

- `dataset_report.txt`
- `dataset_report.json`

Optional arguments:

- `--project-root` : Project root that contains `data/raw/`. Defaults to the current working directory.
- `--dataset` : Dataset folder name in `data/raw/` (e.g., `financial_fraud_detection`). Required.
- `--nrows` : Read only the first N rows of the file. Defaults to all rows. Use it for a quick trial run only, because the file is not time-ordered and a partial read distorts the per-account sequence statistics.

### 3. Probe the Label Signal

Train a strong supervised model on a chronological split and compare it with a shuffled-label control. If the best supervised model cannot beat chance, no model can detect fraud in this dataset.

```Shell
uv run python -m lstm_autoencoder probe-signal --dataset financial_fraud_detection --max-train-rows 1000000
```

On PowerShell, replace the trailing `\` with a backtick (`` ` ``) or put the command on one line.

The probe saves the following artifact under `artifacts/`:

- `signal_report.txt`

The report ends with one of three verdicts:

- No signal: test AUC below 0.51 and average precision below 1.15 times the prevalence
- Weak signal: test AUC below 0.60
- Learnable: test AUC of 0.60 or higher

Optional arguments:

- `--project-root` : Project root that contains `data/raw/`. Defaults to the current working directory.
- `--dataset` : Dataset folder name in `data/raw/` (e.g., `financial_fraud_detection`). Required.
- `--nrows` : Read only the first N rows of the file. Defaults to all rows. Same caveat as `inspect-data`.
- `--max-train-rows` : Maximum number of training rows sampled for the probe model. Defaults to 1,500,000. Lower it to speed up the run. Validation and test rows are never subsampled.

_Last Updated 4 November 2026_