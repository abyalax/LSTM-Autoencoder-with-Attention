Read this before writing code
File Structure
```Shell
└── 📁lstm-autoencoder
    └── 📁artifacts
        └── 📁model
            └── 📁paysim_synthetic_financial
                ├── lstm_ae_w1_best.pt
                ├── lstm_ae_w10_best.pt
                ├── lstm_ae_w3_best.pt
                ├── lstm_ae_w5_best.pt
        └── 📁report
            └── 📁paysim_synthetic_financial
                └── 📁baseline
                    └── 📁BaselineEvaluator
                        └── 📁20261104_145907
                            ├── ablation_report.json
                            └── ablation_report.txt
                └── 📁exploratory
                    └── 📁PaySimSyntheticFinancialDatasetsExplorer
                        └── 📁20261104_100038
                            ├── dataset_report.json
                            ├── dataset_report.txt
                        └── 📁20261104_100136
                            ├── dataset_report.json
                            └── dataset_report.txt
                └── 📁model_training
                    └── 📁PaySimModelTrainingPipeline
                        └── 📁20261104_164946
                            ├── training_report.json
                            ├── training_report.txt
                        └── 📁20261104_165221
                            ├── training_report.json
                            └── training_report.txt
        └── 📁temp
            ├── temporal_drift_data.csv
    └── 📁data
        └── 📁raw
            └── 📁financial_fraud_detection
                ├── financial_fraud_detection.csv
                └── financial_fraud_detection.zip
            └── 📁paysim_synthetic_financial
                ├── 📁paysim_synthetic_financial.csv
                └── 📁paysim_synthetic_financial.zip
        └── 📁processed
    └── 📁scripts
    └── 📁src
        └── 📁lstm_autoencoder
            └── 📁data
                └── 📁financial_fraud_detection
                    ├── __init__.py
                    ├── causal_features.py
                    ├── exploratory.py
                    ├── schema.py
                    ├── sequence_profile.py
                └── 📁paysim_synthetic_financial
                    ├── __init__.py
                    ├── causal_features.py
                    ├── exploratory.py
                    ├── schema.py
                    ├── sequence_profile.py
                ├── __init__.py
                ├── loader.py
            └── 📁evaluation
                └── 📁financial_fraud_detection
                    ├── __init__.py
                    ├── metrices.py
                    ├── signal_probe.py
                └── 📁paysim_synthetic_financial
                    ├── __init__.py
                    ├── baseline.py
                    ├── metrics.py
                    ├── signal_probe.py
                ├── __init__.py
                ├── metrics.py
                ├── signal_probe.py
            └── 📁model
                ├── __init__.py
                ├── config.py
                ├── dataset.py
                ├── feature_scaler.py
                ├── lstm_autoencoder.py
                ├── pipeline.py
                ├── sequence_builder.py
                ├── trainer.py
            └── 📁reporting
                ├── __init__.py
                ├── console_report.py
                ├── json_report.py
                ├── reporter.py
            ├── __init__.py
            ├── __main__.py
            ├── cli.py
            └── config.py
    ├── .gitignore
    ├── .python-version
    ├── pyproject.toml
    ├── README.md
    └── uv.lock
```
Principal Design
- Single Responsibility: setiap class memiliki tanggung jawab yang jelas.
- One Main Class per Module: setiap file Python memiliki satu class utama. Pengecualian hanya untuk dataclass, enum, dan tipe data pendukung.
- Explicit Typing: semua parameter dan return value function/method memiliki type annotation.
- Private Helpers: fungsi internal menggunakan prefix _, misalnya _normalize_target().
- CLI as Entry Point: cli.py hanya mengurus argument parsing dan dispatch command, bukan menjalankan proses analisis secara langsung.
- Domain Separation: data loading, exploration, metrics, dan reporting dipisahkan.
- class biasa: memiliki behavior atau proses.
- @dataclass(frozen=True): merepresentasikan data yang tidak perlu dimodifikasi.
- Enum: merepresentasikan kumpulan nilai konstan.
- Private methods: implementasi internal suatu class.
- Public methods: behavior yang memang boleh digunakan modul lain.
- Tidak perlu membuat dataclass untuk setiap variabel atau hasil operasi sederhana.
- Hindari business logic di CLI.
- Hindari class yang terlalu besar.
- Hindari membuat utility module yang terlalu umum.
- Jangan langsung membuat utils.py yang berisi berbagai macam function tanpa hubungan domain.
- Jangan memaksakan OOP.
- Tidak semua hal harus menjadi class. 
- Function internal yang sederhana boleh menjadi private method. 
- Dataclass hanya digunakan ketika memang ada kebutuhan merepresentasikan data.
- Pisahkan konfigurasi dan data.
- Path dataset, nama kolom target, parameter model, dan konfigurasi training melalui configuration object
- output long execution must be report and save on file 
- Avoid consecutive if statements for mutually exclusive or priority-based conditions; prefer dictionary mapping or match-case where appropriate.
- Preserve condition precedence and fallback behavior while prioritizing readability over unnecessary complexity.

Flow Data and Execution (example)
CLI
 │
 └── inspect-data
       │
       └── DatasetExplorer
              │
              ├── DatasetLoader
              │      └── load()
              │
              └── ConsoleReporter
                     ├── display_section()
                     ├── display_text()
                     └── display_table()

| Aturan             | Konvensi                            |
| ------------------ | ----------------------------------- |
| Class utama        | `PascalCase`                        |
| Dataclass          | `PascalCase`                        |
| Public method      | `snake_case`                        |
| Private method     | `_snake_case`                       |
| Constant           | `UPPER_SNAKE_CASE`                  |
| Function arguments | Wajib type annotation               |
| Return value       | Wajib type annotation               |
| Class attributes   | Wajib type annotation               |
| Docstring          | Wajib untuk public class dan method |
| Imports            | Absolute imports                    |
| Formatting         | Ruff                                |
| Testing            | Pytest                              |