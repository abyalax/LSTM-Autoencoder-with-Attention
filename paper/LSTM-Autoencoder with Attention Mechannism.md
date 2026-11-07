# A. Judul
**Deteksi Anomali Transaksi Keuangan Menggunakan Hybrid LSTM-Autoencoder dengan Attention Mechanism**
# B. Phenomenon
Pertumbuhan transaksi keuangan digital telah meningkatkan volume dan kompleksitas aliran dana, sekaligus memperluas peluang penyalahgunaan akun sebagai perantara pemindahan dana. Salah satu pola yang relevan untuk dikaji adalah aktivitas akun yang menerima dana melalui transaksi `TRANSFER` dan kemudian melakukan pencairan melalui `CASH_OUT`. Pola tersebut dapat mengindikasikan karakteristik akun perantara (money mule), sehingga diperlukan pendekatan yang mampu menganalisis rangkaian aktivitas transaksi berdasarkan identitas akun dan urutan waktunya.
Deteksi fraud menghadapi tantangan berupa ketidakseimbangan kelas (class imbalance) yang ekstrem, keterbatasan ketersediaan label, serta perubahan pola aktivitas transaksi. Dalam kondisi tersebut, pendekatan yang sepenuhnya bergantung pada klasifikasi berbasis label memiliki keterbatasan dalam mengidentifikasi pola anomali yang belum diketahui sebelumnya. Autoencoder dapat digunakan untuk mempelajari representasi pola aktivitas normal dan mengidentifikasi penyimpangan berdasarkan kesalahan rekonstruksi (reconstruction error).
Berdasarkan permasalahan tersebut, penelitian ini mengusulkan Hybrid LSTM-Autoencoder dengan Attention Mechanism untuk mempelajari pola temporal aktivitas akun penerima dan mengidentifikasi anomali pada rangkaian transaksi.
# C. Research Gap
## C1. Penelitian LSTM dengan attention umumnya berupa classifier supervised. 
Penelitian Benchaji dkk. menggunakan Attention pada LSTM untuk meningkatkan kemampuan klasifikasi fraud. Marco dkk. menggunakan CNN-LSTM-Attention dengan SMOTE untuk mengatasi ketidakseimbangan kelas. Pendekatan tersebut masih bergantung pada label dan proses penyeimbangan kelas.
https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00541-8
https://inass.org/wp-content/uploads/2024/12/2025043045-2.pdf
## C2. Pendekatan autoencoder dan LSTM-AE belum mengintegrasikan attention
Tesis Kolawole (NCI 2023) memakai arsitektur autoencoder-LSTM bersama SMOTE. Almuteer dkk. membandingkan CNN, AE, LSTM dan AE+LSTM, dan model AE+LSTM justru memperoleh akurasi terendah (0,32) dibanding AE saja (0,99). Menggabungkan komponen secara naif tidak otomatis lebih baik.
https://norma.ncirl.ie/7550/
https://igi.indrastra.com/items/show/2167
## C3. Sebuah Paper di FUDMA Journal of Sciences menyebut studi yang menggabungkan XGBoost, attention, dan LSTM-AE pada dataset IEEE-CIS. 
Pipeline dengan classifier akhir, bukan attention didalam model rekonstruksi. Kontribusi attention secara terisolasi (ablation) pada LSTM-AE unsupervised belum jelas dikaji.
https://fjs.fudutsinma.edu.ng/index.php/fjs/article/download/3072/2236
## C4. Evaluasi real-time jarang dibahas explisit.
Achituve dkk. memang menargetkan deteksi real-time pada online banking dan menekankan interpretability, tetapi memakai classifier, bukan reconstruction-based.
https://u.cs.biu.ac.il/~sarit/data/articles/Interpretable_Online_Banking_Fraud.pdf
## C5. Paper FUDMA yang sama mencatat concept drift tidak ditangani.
Keterbatasan penelitian, (hanya diakui, tidak harus dijadikan target)
https://fjs.fudutsinma.edu.ng/index.php/fjs/article/download/3072/2236
# D. Research Question
## D1. LSTM-Autoencoder dibanding baseline
Seberapa baik LSTM-Autoencoder dengan attention mendeteksi transaksi fraud dibanding baseline (Isolation Forest, Autoencoder basic, LSTM-AE tanpa attention, LSTM Classifier)?
Diukur dengan PR-AUC, Recall pada FPR tetap, Precision, F1-score, dan MCC.
## D2. Kontribusi attention pada LSTM-Autoencoder
Seberapa besar kontribusi attention terhadap performa, dan apakah bobot attention menyorot transaksi atau fitur yang bermakna dalam sekuens akun?
## D3. Pengaruh Window dan History Length
Bagaimana pengaruh panjang window dan variasi panjang riwayat akun terhadap performa deteksi anomali?
Grid Experiments :  $W \ in \ {1, 3, 5, 10}$

# E. Urgensi Penelitian
## E1. Pertumbuhan transaksi digital meningkatkan risiko penyalahgunaan akun perantara
Pertumbuhan transaksi keuangan digital meningkatkan volume dan kompleksitas aliran dana, sekaligus memperluas peluang penyalahgunaan akun sebagai perantara pemindahan dana. Aktivitas tersebut dapat tercermin melalui pola penerimaan dana (`TRANSFER`) yang diikuti pencairan (`CASH_OUT`). Oleh karena itu, diperlukan mekanisme deteksi yang mampu mengidentifikasi anomali temporal pada aktivitas akun penerima.
## E2. Ketidakseimbangan data dan keterbatasan label menjadi tantangan deteksi anomali akun
Verma dan Dhar mengidentifikasi class imbalance, keterbatasan label, dan verification latency sebagai tantangan dalam sistem deteksi fraud. Autoencoder menawarkan pendekatan unsupervised anomaly detection dengan mempelajari pola normal aktivitas akun penerima untuk mengidentifikasi penyimpangan tanpa bergantung sepenuhnya pada label fraud.
https://arxiv.org/abs/2409.13406
## E3. Pola penerimaan dan pencairan dana membutuhkan analisis temporal berbasis akun
Ding dkk. memandang deteksi fraud sebagai permasalahan anomaly detection. Perspektif ini mendukung analisis aktivitas akun berdasarkan urutan transaksi, khususnya pola penerimaan dana melalui `TRANSFER` dan pencairan melalui `CASH_OUT`. Dengan menggunakan identitas akun sebagai unit analisis, deteksi diarahkan pada rangkaian aktivitas akun, bukan karakteristik transaksi individual semata.
https://peerj.com/articles/cs-2323/
## E4. Attention membantu mengidentifikasi pola transaksi yang relevan
Mekanisme Attention memungkinkan model memberikan bobot berbeda terhadap timestep dalam rangkaian transaksi, sehingga bagian aktivitas akun yang paling berpengaruh terhadap rekonstruksi dapat lebih diperhatikan. Hal ini mendukung identifikasi pola temporal yang mencurigakan sekaligus memberikan informasi tambahan untuk interpretasi hasil deteksi.
## E5. Variasi panjang riwayat akun memengaruhi evaluasi deteksi anomali
Akun penerima memiliki jumlah riwayat transaksi yang bervariasi. Akun dengan riwayat pendek berpotensi menghasilkan representasi temporal yang berbeda dibandingkan akun dengan riwayat panjang. Oleh karena itu, evaluasi perlu mempertimbangkan stratifikasi berdasarkan panjang riwayat (history bucket) untuk mengetahui konsistensi performa model pada berbagai kondisi ketersediaan riwayat transaksi.
# F. State Of The Art
## F1. LSTM dan Attention (supervised).
Memodelkan sekuens transaksi, dengan attention untuk fokus pada elemen penting (Benchaji 2021, Achituve 2019).
https://doi.org/10.1186/s40537-021-00541-8
https://doi.org/10.1109/MLSP.2019.8918896
## F2. Autoencoder dan LSTM-AE (unsupervised/semi-supervised).
Mendeteksi anomali dari reconstruction error, sering dikombinasikan dengan resampling atau classifier lain (Kolawole 2023, Ding 2024).
https://norma.ncirl.ie/7550/
https://peerj.com/articles/cs-2323/
## F3. Hybrid Kompleks
Menggabungkan ResNet, LSTM dan attention, umumnya di dataset kartu kredit seperti IEEE-CIS (Umaru 2025).
https://fjs.fudutsinma.org.ng/index.php/fjs/article/view/3072
# G. Novelty
Account-centric temporal anomaly detection menggunakan Hybrid LSTM-Autoencoder dengan Attention Mechanism.
Kontribusi penelitian:
1. Recipient-centric sequence construction: membangun sekuens berdasarkan aktivitas akun penerima dan pencairan dana.
2. Attention integration: mengintegrasikan Attention sebagai pooling pada bottleneck LSTM-Autoencoder.
3. Attention ablation: menguji kontribusi Attention terhadap kemampuan rekonstruksi dan deteksi anomali.
4. History-length evaluation: mengevaluasi performa berdasarkan variasi panjang riwayat akun.
# H. [[College/Research/LSTM-Autoencoder/Literature Review/Literature Review|Literature Review]]
# I. Methodology
Long Short-Term Memory Auto-Encoder with Attention Mechanism

| Komponen          | Keputusan final                                      |
| ----------------- | ---------------------------------------------------- |
| Dataset utama     | PaySim                                               |
| Unit analisis     | Akun penerima/perantara                              |
| Identitas akun    | `nameDest` untuk TRANSFER; `nameOrig` untuk CASH_OUT |
| Transaction scope | TRANSFER dan CASH_OUT                                |
| Sequence          | Riwayat aktivitas akun berdasarkan waktu             |
| Window grid       | `{1, 3, 5, 10}`                                      |
| History handling  | Padding, mask, dan history length                    |
| Model             | LSTM-Autoencoder + Attention                         |
| Training          | Window normal pada training set                      |
| Scoring           | Reconstruction error                                 |
| Threshold         | Ditentukan menggunakan validation set                |
| Inference         | Causal sliding window                                |
Catatan konstruksi akun: untuk menghubungkan `TRANSFER → CASH_OUT`, identitas akun harus disatukan berdasarkan peran akun yang sedang dianalisis. Pada TRANSFER, akun fokus berasal dari `nameDest`; pada CASH_OUT, akun fokus berasal dari `nameOrig`.
# J. Baseline
Pembanding dibagi tiga tingkat dengan fitur, split, dan anggaran tuning yang sama.

| Tingkat              | Model                                                              | Tujuan                                         |
| -------------------- | ------------------------------------------------------------------ | ---------------------------------------------- |
| Klasik unsupervised  | Isolation Forest, PCA reconstruction error                         | Batas bawah non-deep                           |
| Deep unsupervised    | Dense Autoencoder; LSTM-AE tanpa attention                         | Menguji sumbangan sekuens dan attention        |
| Referensi supervised | LightGBM/XGBoost; LSTM + attention classifier (gaya Benchaji dkk.) | Batas atas berlabel, bukan perbandingan setara |
# K. Proposed Model
LSTM-Autoencoder dengan attention-pooling pada bottleneck.
- **Encoder.** Satu atau dua lapis LSTM menghasilkan hidden state `h_1 … h_W`.
- **Attention.** Attention aditif memberi bobot `α_t` pada tiap `h_t`, lalu context vector `c = Σ α_t h_t`.
- **Bottleneck.** Layer linear memetakan `c` ke `z` berdimensi kecil (misal 16–64).
- **Decoder.** `z` diulang sebanyak W langkah, diteruskan ke LSTM decoder, lalu layer output merekonstruksi window.
- **Loss.** MSE untuk fitur numerik, ditambah cross-entropy untuk fitur kategorikal jika memakai embedding.
Ada satu keputusan desain yang perlu diwaspadai. Kalau decoder boleh melakukan attention ke seluruh state encoder (cross-attention), jalur itu melewati bottleneck dan model bisa belajar menyalin input. Akibatnya anomali pun direkonstruksi dengan baik dan error tidak lagi membedakan fraud. Karena itu usulan utama memakai attention sebagai pooling menuju bottleneck tunggal, dan versi cross-attention dijadikan varian ablation untuk menunjukkan masalah tersebut. Ini sekaligus menjawab kehati-hatian dari temuan Almuteer dkk. bahwa gabungan AE dan LSTM tidak otomatis lebih baik.
# L. Experiments / Ablation
Setiap konfigurasi dijalankan dengan 5 seed, dilaporkan sebagai rata-rata ± simpangan baku, dan evaluasi akhir memakai test set yang tidak pernah disentuh saat tuning. Grid kecil: $W ∈ {5, 10, 20}$,
$dimensi laten ∈ {16, 32, 64}$, $hidden ∈ {64, 128}$.

| Kode | Eksperimen                                                                                                                                               | Menjawab             |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| E1   | Perbandingan utama: semua baseline vs model usulan                                                                                                       | RQ1                  |
| E2   | Ablation attention: tanpa attention (hidden state terakhir), mean-pooling, attention aditif (usulan), multi-head self-attention, cross-attention decoder | RQ2                  |
| E3   | Panjang window dan strategi threshold (persentil-99 error normal di train vs dipilih dari val berlabel)                                                  | RQ3                  |
| E4   | Latensi dan throughput per transaksi                                                                                                                     | RQ4                  |
| E5   | Ablation fitur: tanpa fitur relasional, tanpa fitur selisih waktu, dengan vs tanpa kolom berisiko leakage                                                | Kontribusi fitur     |
| E6   | Analisis per segmen (transaction_type, device, periode waktu)                                                                                            | Robustness dan drift |
| E7   | Visualisasi bobot attention pada window fraud vs normal                                                                                                  | Interpretabilitas    |
# M. Evaluation
Dengan prevalensi 4%, akurasi menyesatkan karena model yang selalu menjawab “normal” sudah mencapai 96%. Metrik yang dilaporkan:
- **AUPRC** sebagai metrik utama. Baseline acaknya sama dengan prevalensi (0,04).
- **ROC-AUC** sebagai pelengkap.
- **Recall pada FPR tetap** (1% dan 0,1%) dan **precision@k**, yang mencerminkan kapasitas tim investigasi memeriksa alert.
- **Precision, recall, F1, dan MCC** pada threshold yang dipilih di validasi.
- **Interval kepercayaan bootstrap** pada test set untuk perbandingan antar model.
- **Latensi inferensi per transaksi** (p50, p95, p99) dan throughput, pada CPU dan GPU. Definisi “real-time” perlu Anda tetapkan sebagai target operasional di proposal, misalnya p95 di bawah 100 ms. Angka ini asumsi saya, bukan standar baku, jadi sesuaikan dengan referensi yang Anda pakai.
- **Uji sensitivitas prevalensi.** 4% jauh di atas fraud dunia nyata, jadi ulangi evaluasi dengan fraud di test di-subsample ke sekitar 0,5–1%. Ini menunjukkan seberapa turun precision pada kondisi realistis.
# N. Dataset

[Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1/data)
> _Synthetic datasets generated by the PaySim mobile money simulator_

| Column         | Keterangan                                                                    |
| -------------- | ----------------------------------------------------------------------------- |
| step           | Maps a unit of time in the real world. In this case 1 step is 1 hour of time. |
| type           | CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER                                |
| amount         | amount of the transaction in local currency                                   |
| nameOrig       | customer who started the transaction                                          |
| oldbalanceOrg  | initial balance before the transaction                                        |
| newbalanceOrig | customer's balance after the transaction.                                     |
| nameDest       | recipient ID of the transaction.                                              |
| oldbalanceDest | recipient's balance before the transaction.                                   |
| newbalanceDest | recipient's balance after the transaction.                                    |
| isFraud        | identifies a fraudulent transaction (1) and non fraudulent (0)                |
Scope transaksi dibatasi pada:

- `TRANSFER`
- `CASH_OUT`

Kolom yang dikecualikan:

- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`
- `isFlaggedFraud` sebagai fitur.

`isFraud` digunakan sebagai label evaluasi, bukan sebagai input model.

Batasan penelitian:

- Dataset bersifat sintetis.
- Label fraud merupakan label transaksi, bukan ground truth akun money mule.
- Periode simulasi sekitar 31 hari.
- Hasil tidak dapat digeneralisasikan sebagai bukti identifikasi pelaku pencucian uang.
- Klaim concept drift jangka panjang tidak menjadi target penelitian.

[Financial Transaction Datasets for Fraud Detection](https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection/data)
> _Large-scale synthetic dataset for training and fraud detection model._

| Column            | Keterangan                                                               |
| ----------------- | ------------------------------------------------------------------------ |
| transaction_id    | Unique identifier for each transaction.                                  |
| timestamp         | ISO-format timestamp when the transaction occurred.                      |
| sender_account    | Account ID of the transaction initiator                                  |
| receiver_account  | Account ID of the transaction recipient.                                 |
| amount            | Transaction amoun in USD                                                 |
| transaction_type  | Type of transaction: deposit, withdrawal, transfer, or payment.          |
| merchant_category | Business category involved in the transaction (e.g., retail, utilities). |
| location          | Location from where the transaction was initiated.                       |
| device_used       | Device type used: mobile, web, atm, pos.                                 |
| is_fraud          | Boolean flag indicating whether the transaction was fraudulent.          |

# Writing
## BAB I — Pendahuluan
_Bab ini menjawab: Mengapa penelitian ini perlu dilakukan?

| Poin                      | Isi yang harus dibahas                                                                              |
| ------------------------- | --------------------------------------------------------------------------------------------------- |
| 1.1 Latar Belakang        | Fenomena fraud transaksi keuangan, pertumbuhan transaksi digital, dan tantangan mendeteksi anomali. |
| 1.2 Identifikasi Masalah  | Permasalahan yang ditemukan pada sistem deteksi anomali konvensional atau pendekatan terdahulu.     |
| 1.3 Rumusan Masalah       | Pertanyaan spesifik yang akan dijawab penelitian.                                                   |
| 1.4 Batasan Masalah       | Batasan dataset, jenis anomali, arsitektur, dan lingkungan eksperimen.                              |
| 1.5 Tujuan Penelitian     | Tujuan yang ingin dicapai.                                                                          |
| 1.6 Manfaat Penelitian    | Manfaat teoretis dan praktis.                                                                       |
| 1.7 Sistematika Penulisan | Penjelasan struktur setiap bab.                                                                     |
| TTTTTTTTTTTTTTTTTTTTT     | _just for tweak formatting (ignore this)                                                            |

## BAB II — Tinjauan Pustaka dan Landasan Teori
_Bab ini menjawab: Apa dasar ilmiah dan teknologi yang digunakan?

| Poin                          | Isi yang harus dibahas                                             |
| ----------------------------- | ------------------------------------------------------------------ |
| 2.1 Penelitian Terdahulu      | Artikel jurnal dan penelitian relevan tentang anomaly detection.   |
| 2.2 Fraud Detection           | Konsep penipuan dan anomali dalam transaksi keuangan.              |
| 2.3 Anomaly Detection         | Konsep deteksi anomali dan pendekatan-pendekatannya.               |
| 2.4 Deep Learning             | Konsep dasar Deep Learning.                                        |
| 2.5 Artificial Neural Network | Dasar jaringan saraf tiruan.                                       |
| 2.6 Recurrent Neural Network  | Konsep pemrosesan data sekuensial.                                 |
| 2.7 Long Short-Term Memory    | Arsitektur dan mekanisme LSTM.                                     |
| 2.8 Autoencoder               | Encoder, latent representation, decoder, dan reconstruction error. |
| 2.9 Attention Mechanism       | Konsep attention dan perhitungan attention weights.                |
| 2.10 Hybrid LSTM Autoencoder  | Integrasi komponen LSTM dan Autoencoder.                           |
| 2.11 Evaluation Metrics       | Precision, Recall, F1-score, ROC-AUC, PR-AUC, dan metrik lainnya.  |
| 2.12 Real-Time Processing     | Konsep streaming, latency, throughput, dan pemrosesan transaksi.   |
| 2.13 Kerangka Pemikiran       | Hubungan antara permasalahan, metode, dan hasil yang diharapkan.   |
| TTTTTTTTTTTTTTTTTTTTTTTTTT    | _just for tweak formatting (ignore this)                           |
## BAB III — Metodologi Penelitian
_Bab ini menjawab: Bagaimana penelitian dilaksanakan?

| Poin                          | Isi yang harus dibahas                                               |
| ----------------------------- | -------------------------------------------------------------------- |
| 3.1 Jenis Penelitian          | Penelitian laboratorium dengan pendekatan kuantitatif eksperimental. |
| 3.2 Tahapan Penelitian        | Urutan proses penelitian dari awal sampai akhir.                     |
| 3.3 Dataset                   | Sumber, karakteristik, jumlah data, fitur, dan label.                |
| 3.4 Exploratory Data Analysis | Analisis distribusi, korelasi, missing values, dan imbalance.        |
| 3.5 Data Preprocessing        | Cleaning, encoding, normalisasi, dan transformasi data.              |
| 3.6 Sequence Construction     | Membentuk urutan transaksi untuk input LSTM.                         |
| 3.7 Dataset Splitting         | Pembagian data training, validation, dan testing.                    |
| 3.8 Perancangan Arsitektur    | Detail arsitektur Hybrid LSTM-Autoencoder dengan Attention.          |
| 3.9 Model Training            | Konfigurasi training, optimizer, loss function, epoch, batch size.   |
| 3.10 Anomaly Threshold        | Penentuan ambang reconstruction error.                               |
| 3.11 Simulasi Real-Time       | Mekanisme pemrosesan transaksi secara berkelanjutan.                 |
| 3.12 Skenario Eksperimen      | Pengujian model dan pembanding.                                      |
| 3.13 Evaluasi                 | Metrik evaluasi performa deteksi dan performa pemrosesan.            |
| TTTTTTTTTTTTTTTTTTTTTTTT      | _just for tweak formatting (ignore this)                             |
> Note
> Technical Documentation 
- Input layer.
- LSTM encoder.
- Hidden representation.
- Attention layer.
- LSTM decoder.
- Reconstruction output.
- Reconstruction loss.
- Anomaly threshold.
- Output klasifikasi normal atau anomali.

Skenario Experimen

| Eksperimen | Model                                |
| ---------- | ------------------------------------ |
| E1         | Vanilla Autoencoder                  |
| E2         | LSTM-Autoencoder                     |
| E3         | LSTM-Autoencoder + Attention         |
| E4         | Isolation Forest (baseline tambahan) |

Realtime Skenario Test
- **Latency**: waktu yang diperlukan untuk memproses satu transaksi.
- **Throughput**: jumlah transaksi yang dapat diproses per satuan waktu.
- **P95 Latency**: batas waktu pemrosesan yang mencakup 95
- **Resource Utilization**: penggunaan CPU dan RAM selama pemrosesan.
## BAB IV — Hasil dan Pembahasan
_Bab ini menjawab: Apa yang ditemukan dari eksperimen?

| Poin                    | Isi yang harus dibahas                                        |
| ----------------------- | ------------------------------------------------------------- |
| 4.1 Hasil Preprocessing | Kondisi dataset setelah preprocessing.                        |
| 4.2 Hasil Training      | Training loss, validation loss, dan convergence model.        |
| 4.3 Hasil Pengujian     | Hasil prediksi pada data testing.                             |
| 4.4 Confusion Matrix    | Distribusi prediksi benar dan salah.                          |
| 4.5 Evaluasi Model      | Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC.       |
| 4.6 Perbandingan Model  | Perbandingan model utama dengan baseline.                     |
| 4.7 Ablation Study      | Analisis kontribusi Attention Mechanism.                      |
| 4.8 Evaluasi Real-Time  | Latency, throughput, dan penggunaan sumber daya.              |
| 4.9 Pembahasan          | Interpretasi hasil dan kaitannya dengan penelitian terdahulu. |
Evaluation
- Precision: seberapa banyak prediksi fraud yang benar.
- Recall: seberapa banyak fraud yang berhasil ditemukan.
- F1-score: keseimbangan Precision dan Recall.
- PR-AUC: performa pada berbagai threshold dengan perhatian pada kelas positif.
Be Aware with Data Leakage
Confusion Matrix
## BAB V — Penutup
_Bab ini menjawab: Apa kesimpulan penelitian dan apa yang masih perlu dikembangkan?

| Poin                        | Isi yang harus dibahas                                                 |
| --------------------------- | ---------------------------------------------------------------------- |
| 5.1 Kesimpulan              | Jawaban terhadap seluruh rumusan masalah berdasarkan hasil eksperimen. |
| 5.2 Keterbatasan Penelitian | Keterbatasan dataset, arsitektur, eksperimen, dan simulasi.            |
| 5.3 Saran                   | Rekomendasi untuk penelitian selanjutnya.                              |
| TTTTTTTTTTTTTTTTTTTTTT      | _just for tweak formatting (ignore this)                               |
