# Deteksi Anomali Transaksi Keuangan Menggunakan Hybrid LSTM-Autoencoder dengan Attention Mechanism

*[Judul kerja, tanpa klaim real-time. Penulis dan afiliasi: isi.]*

> **Catatan penyusunan (hapus sebelum submit).** Semua angka diambil dari `artifacts/summary/results_summary.md` dan report JSON di commit `d0a5e02` (3 seed: 42, 123, 2026), setelah retrain penuh (semua arsitektur x $W$) dengan early stopping berbasis reconstruction loss. Draf sengaja melaporkan hasil apa adanya: fitur LSTM-AE memberi gain kecil dan tidak signifikan di AUC.
>
> Pembaruan 2026-11-07: (1) Bug `hybrid_val_norm` telah diperbaiki untuk menggunakan normalisasi leak-free dengan statistik train-normal; (2) `_encode` kini deterministik (sorted) untuk reproduktibilitas; (3) Bootstrap CIs untuk recall@FPR 1% dan 0,1% telah ditambahkan; (4) Tabel per-bucket side-by-side telah ditambahkan ke `summarize-results`; (5) `no_history` kini juga membuang dummy `history_bucket_*` (sebelumnya bocor via jalur itu), `account_code` dikeluarkan dari fitur LR/GBM, dan `amount`/`history_length` di-log1p sebelum standardisasi; (6) arsitektur `attn_pool` (attention pooling ke satu vektor bottleneck) diimplementasikan dan diuji; (7) early stopping LSTM-AE diganti dari val AUC ke val reconstruction loss (normal-only) untuk menutup kebocoran label pada model selection, lalu seluruh konfigurasi dilatih ulang.
>
> Pembaruan 2026-11-08 (investigasi "identity collapse", lihat bagian 4.5): dicoba bottleneck berdimensi sangat kecil ($z \in \{2,3,4\}$, hidden=16) dengan harapan memaksa kompresi nyata. Dengan seleksi checkpoint yang valid (val reconstruction loss, bukan val AUC), ketiga varian tetap runtuh ke AUC uji 0,42–0,50 pada epoch-epoch akhir — tidak lebih baik dari arsitektur sebelumnya. Satu titik anomali ditemukan: AUC validasi pada varian $z=3$ memuncak di epoch 1 (0,8023) lalu menurun terus hingga 0,4357 di epoch terpilih (loss-based); bila checkpoint sengaja dipilih dari AUC validasi (bukan loss), angka uji mencapai 0,8070. Angka ini **tidak dilaporkan sebagai temuan positif**: ia berasal dari satu seed, satu epoch nyaris tanpa pelatihan, dan dipilih memakai label validasi, persis kebocoran model-selection yang didokumentasikan sebagai risiko di draf sebelumnya. Investigasi ini dianggap memperkuat, bukan membantah, kesimpulan bahwa pendekatan rekonstruksi sudah mentok untuk data ini.

## Abstrak

Akun perantara yang menerima lalu mencairkan dana (pola TRANSFER diikuti CASH_OUT) sulit dideteksi karena label fraud langka dan riwayat akun penerima umumnya sangat pendek. Penelitian ini menguji apakah *reconstruction error* LSTM-Autoencoder, dengan beberapa varian attention, menambah kemampuan deteksi pada dataset sintetis PaySim (dibatasi pada transaksi CASH_OUT dan TRANSFER (2.770.409 transaksi, prevalensi fraud 0,30%)). Sekuens dibangun per akun penerima dengan window $W \in \{1,3,5,10\}$, padding, dan mask. Empat arsitektur dibandingkan (plain, bottleneck, self-attention bermask diagonal, attention pooling) bersama baseline Logistic Regression, gradient boosting, Isolation Forest, dan dense autoencoder. Sebagai detektor mandiri, LSTM-AE hanya mencapai AUC 0,49–0,67 dan kalah dari semua baseline berlabel maupun Isolation Forest (AUC 0,8665). Sebagai fitur tambahan pada Logistic Regression dengan protokol bebas leakage, error AE menaikkan AP sekitar 4–6% secara relatif (+0,0017 sampai +0,0024) tetapi tidak menaikkan AUC secara signifikan (CI bootstrap di atas nol pada 0 dari 3 seed di seluruh 16 konfigurasi), dan hasilnya tidak bergantung pada varian attention. Panjang riwayat akun penerima adalah sinyal terkuat: menghapusnya menurunkan AUC dari 0,8661 ke 0,8027. Hasil ini menunjukkan batas pendekatan berbasis rekonstruksi pada data dengan riwayat akun tipis.

**Kata kunci:** deteksi anomali, LSTM-Autoencoder, attention, fraud keuangan, PaySim, akun penerima.

## 1. Pendahuluan

Pertumbuhan transaksi digital memperluas peluang penyalahgunaan akun sebagai perantara pemindahan dana. Pola yang sering muncul adalah akun menerima dana lewat TRANSFER lalu segera mencairkannya lewat CASH_OUT, sehingga unit analisis yang bermakna adalah aktivitas akun, bukan karakteristik transaksi tunggal ([Ding dkk.](https://peerj.com/articles/cs-2323/)). Deteksi fraud menghadapi ketidakseimbangan kelas, keterbatasan label, dan keterlambatan verifikasi ([Verma dan Dhar](https://arxiv.org/abs/2409.13406)); autoencoder menawarkan pendekatan tanpa label yang mempelajari pola normal lalu menandai penyimpangan lewat *reconstruction error*.

Kajian terdahulu memperlihatkan celah berikut. **(C1)** LSTM dengan attention umumnya dipakai sebagai *classifier* supervised yang bergantung pada label dan penyeimbangan kelas ([Benchaji dkk.](https://doi.org/10.1186/s40537-021-00541-8)). **(C2)** Autoencoder dan LSTM-AE belum diintegrasikan dengan attention, dan penggabungan naif tidak otomatis lebih baik ([Almuteer dkk.](https://igi.indrastra.com/items/show/2167)). **(C3)** Studi hybrid XGBoost, attention, dan LSTM-AE memakai *classifier* akhir sehingga sumbangan attention di dalam model rekonstruksi tidak terisolasi ([Umaru dkk.](https://fjs.fudutsinma.edu.ng/index.php/fjs/article/view/3072)). **(C4)** *Concept drift* umumnya hanya diakui sebagai keterbatasan.

Penelitian ini menjawab pertanyaan berikut:

- **RQ1.** Seberapa baik LSTM-AE (dengan dan tanpa attention) mendeteksi fraud dibanding baseline Logistic Regression, gradient boosting, Isolation Forest, dan dense autoencoder?
- **RQ2.** Seberapa besar kontribusi attention terhadap performa? **[TODO: bagian "apakah bobot attention menyorot transaksi bermakna" belum dijawab karena ekstraksi bobot attention belum ada di repo.]**
- **RQ3.** Bagaimana pengaruh panjang window dan panjang riwayat akun terhadap performa?

Kontribusi: (1) konstruksi sekuens berpusat pada akun penerima beserta fitur kausal riwayat akun; (2) ablation empat arsitektur LSTM-AE, termasuk attention pooling, yang diisolasi dari *classifier* akhir; (3) evaluasi terstratifikasi menurut panjang riwayat akun; (4) protokol hybrid bebas kebocoran dengan kontrol null (kolom error diacak) dan *paired bootstrap*. Klaim *real-time* dan evaluasi latensi tidak termasuk dalam penelitian ini.

## 2. Tinjauan Pustaka

**LSTM + attention supervised.** Model sekuens dengan attention menyorot elemen penting dalam riwayat transaksi ([Benchaji dkk.](https://doi.org/10.1186/s40537-021-00541-8); [Achituve dkk.](https://doi.org/10.1109/MLSP.2019.8918896)), tetapi memerlukan label dan umumnya memakai *oversampling*.

**Autoencoder dan LSTM-AE.** Deteksi anomali dari *reconstruction error*, sering digabung dengan resampling atau *classifier* lain ([Kolawole](http://norma.ncirl.ie/7550/); [Ding dkk.](https://peerj.com/articles/cs-2323/)). Almuteer dkk. melaporkan AE+LSTM justru terendah di antara kombinasi yang diuji, yang mengingatkan bahwa menambah komponen tidak otomatis menaikkan performa.

**Hybrid kompleks.** Kombinasi ResNet, LSTM, dan attention pada data kartu kredit seperti IEEE-CIS ([Umaru dkk.](https://fjs.fudutsinma.edu.ng/index.php/fjs/article/view/3072)) memakai *classifier* akhir, bukan error rekonstruksi sebagai sinyal terpisah.

**Risiko jalan pintas (*copy shortcut*).** Bila decoder boleh mengakses seluruh keadaan encoder atau attention boleh menunjuk dirinya sendiri, model dapat menyalin input sehingga anomali ikut direkonstruksi baik. Karena itu penelitian ini memakai mask diagonal dan attention sebagai pooling menuju bottleneck tunggal.

## 3. Metode

### 3.1 Dataset dan cakupan

Dataset [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) berisi 6.362.620 transaksi sintetis (8.213 fraud, 0,129%) selama sekitar 31 hari (1 *step* = 1 jam). Fraud hanya muncul pada CASH_OUT dan TRANSFER, sehingga cakupan dibatasi pada kedua tipe itu: 2.770.409 baris dengan prevalensi fraud 0,2965%. Kolom saldo (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) dikeluarkan karena kebocoran (AUC univariat `oldbalanceOrg` = 0,928); `isFraud` hanya dipakai sebagai label evaluasi. Dataset bersifat sintetis dan label fraud adalah label transaksi, bukan *ground truth* akun *money mule*.

### 3.2 Sekuens berpusat pada akun penerima

Windowing per pengirim (`nameOrig`) tidak layak karena median dan persentil ke-95 transaksi per pengirim adalah 1. Sekuens dibangun per penerima (`nameDest`): 509.565 akun unik dengan median riwayat 3 dan persentil ke-95 sebesar 17. Setiap transaksi memakai window berisi paling banyak $W$ transaksi terakhir akun penerima, dengan padding dan mask untuk riwayat yang lebih pendek. Fitur kausal per langkah: `amount`, `step_diff`, `log_amount_ratio`, `history_length`, dan `is_new_receiver`. Akun dikelompokkan ke bucket riwayat {1, 2-4, 5-9, >=10}; bucket 1 memuat 37,8% fraud dengan *lift* 8,34 dibanding transaksi normal, sedangkan bucket >=10 memiliki *lift* 0,41.

Data dipisah dengan stratifikasi per penerima agar akun yang sama tidak bocor antar bagian: latih 1.940.924 baris, validasi 414.442 (1.234 fraud), uji 415.043 (1.234 fraud). Penskalaan fitur (StandardScaler) dipasang hanya pada window normal data latih sebelum windowing.

### 3.3 Arsitektur LSTM-Autoencoder

Encoder berupa LSTM 2 lapis (hidden 64, dropout 0,1). Empat varian dibandingkan:

- **plain:** decoder menerima keluaran encoder per langkah, tanpa attention.
- **bottleneck:** decoder hanya menerima *hidden state* akhir encoder yang diulang sepanjang window.
- **attn:** self-attention dengan mask diagonal (posisi tidak boleh memperhatikan dirinya sendiri, kecuali window hanya punya satu langkah nyata).
- **attn_pool:** attention pooling atas keluaran encoder dengan bobot $\alpha_t$ (softmax atas langkah non-padding). Query adalah vektor yang dapat dipelajari, sedangkan key dan value adalah proyeksi linear dari encoder outputs:

$$
\alpha_t = \frac{\exp(\mathbf{q}^\top \mathbf{K}_t / \sqrt{d})}{\sum_{s} \exp(\mathbf{q}^\top \mathbf{K}_s / \sqrt{d})} \quad \text{(dengan masking padding)}
$$

$$
\mathbf{c} = \sum_{t} \alpha_t \mathbf{V}_t
$$

Model dilatih hanya pada window normal dengan loss MSE. *Early stopping* memakai *reconstruction loss* pada baris normal-only di validasi (bukan AUC validasi): memilih epoch lewat AUC berarti memakai label pada model selection, sehingga tetap merupakan sinyal supervised meskipun pelatihannya sendiri unsupervised. AUC validasi tetap dicatat untuk pelaporan, tidak untuk memilih checkpoint. Tiga seed dipakai: 42, 123, 2026. Skor anomali adalah *reconstruction error* rata-rata kuadrat pada langkah non-padding $\mathcal{T}$:

$$
e = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} \lVert \mathbf{x}_t - \hat{\mathbf{x}}_t \rVert_2^2
$$

### 3.4 Baseline

Baseline dievaluasi pada fitur, split, dan data uji yang sama: Logistic Regression (**full**, **no_history** tanpa fitur riwayat, **history_only** hanya dua fitur riwayat), **HistGradientBoosting** (`full_gbm`, rujukan berlabel; bukan perbandingan setara dengan model tanpa label), **Isolation Forest** (semua fitur, dan versi window 5), serta **dense autoencoder** (window 5). LightGBM, XGBoost, PCA, dan *LSTM classifier* tidak diimplementasikan.

### 3.5 Hybrid dan protokol bebas kebocoran

Error AE dipakai sebagai fitur tambahan pada Logistic Regression dengan $\epsilon = 10^{-8}$:

$$
s(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + \gamma \log(e + \epsilon)
$$

Karena AE dilatih pada window normal data latih, error pada data latih terlalu optimistis. Maka varian utama dipasang pada **validasi**:

- `lr_val_control`: LR tanpa error, dipasang pada validasi (pembanding yang mengisolasi efek fitur error).
- `hybrid_val`: LR dengan fitur error, dipasang pada validasi. `hybrid_val_norm`: error dinormalisasi per bucket riwayat.
- `stack_val`: LR dua parameter di atas logit baseline dan $\log e$, dipasang pada validasi.
- `baseline_train`: LR produksi, dipasang pada data latih.
- `hybrid_train_naive`: protokol bocor (dipasang pada data latih), hanya sebagai diagnosis seberapa besar kebocoran menggembungkan hasil.
- `null_*`: sama dengan `hybrid_val` dan `stack_val`, tetapi kolom error diacak; ini menentukan batas bawah derau.

### 3.6 Metrik dan prosedur

Metrik utama adalah AUPRC/*average precision* (AP; baseline acak = prevalensi uji 0,297%), dilengkapi ROC-AUC dan *recall* pada FPR 1% dan 0,1%. Perbandingan berpasangan memakai 200 resampel *bootstrap* (CI 95%) per seed; hasil dilaporkan sebagai rata-rata ± simpangan baku dari 3 seed. Evaluasi juga distratifikasi per bucket riwayat. **[TODO: F1, MCC, precision@k, strategi threshold (persentil-99 error normal vs dipilih dari validasi), dan uji sensitivitas prevalensi belum diimplementasikan.]**

## 4. Hasil

### 4.1 Baseline (uji)

| Model                                   | AUC             | AP              | Recall@FPR 1% |
| --------------------------------------- | --------------- | --------------- | ------------- |
| LR full                                 | 0,8661          | 0,0426          | 0,247         |
| LR no_history                           | 0,8027          | 0,0211          | -             |
| LR history_only                         | 0,7618          | 0,0087          | -             |
| HistGradientBoosting (rujukan berlabel) | 0,8592          | 0,2938          | -             |
| Isolation Forest (semua fitur)          | 0,8665 ± 0,0018 | 0,0495 ± 0,0054 | 0,207         |
| Isolation Forest (window 5)             | 0,8286 ± 0,0024 | 0,0384 ± 0,0016 | 0,208         |
| Dense AE (window 5)                     | 0,3590 ± 0,0102 | 0,0048 ± 0,0001 | 0,015         |

Isolation Forest tanpa label setara dengan LR berlabel dalam AUC. Gradient boosting memiliki AUC sedikit lebih rendah tetapi AP sekitar 6,9 kali LR, yaitu kemampuan peringkat teratas jauh lebih baik. AUC Dense AE di bawah 0,5 berarti error rekonstruksinya justru sedikit lebih rendah pada transaksi fraud; penyebabnya belum ditemukan (lihat Keterbatasan (10)), sehingga baris ini dilaporkan sebagai baseline yang diketahui bermasalah, bukan temuan yang diklaim.

### 4.2 LSTM-AE sebagai detektor mandiri (AUC uji, rata-rata ± std 3 seed)

| Arsitektur           | $W=1$           | $W=3$           | $W=5$           | $W=10$          |
| -------------------- | --------------- | --------------- | --------------- | --------------- |
| plain                | 0,6710 ± 0,0104 | 0,6176 ± 0,0191 | 0,5780 ± 0,0090 | 0,5897 ± 0,0019 |
| bottleneck           | 0,6710 ± 0,0104 | 0,5017 ± 0,0730 | 0,4905 ± 0,0766 | 0,4981 ± 0,0582 |
| attn (mask diagonal) | 0,6567 ± 0,0347 | 0,6000 ± 0,0033 | 0,5991 ± 0,0028 | 0,5916 ± 0,0177 |
| attn_pool            | 0,6004 ± 0,0247 | 0,6082 ± 0,0025 | 0,5934 ± 0,0014 | 0,5830 ± 0,0068 |

Pada $W=1$, plain dan bottleneck identik karena window satu langkah membuat kedua arsitektur ekuivalen. Seluruh varian berada jauh di bawah LR (0,8661) dan Isolation Forest (0,8665). Bottleneck runtuh mendekati acak pada $W \geq 3$. Tidak ada tren monoton terhadap $W$ yang konsisten antar arsitektur; $W=1$ terbaik pada tiga dari empat arsitektur.

### 4.3 Hybrid: kontribusi fitur error AE

Pada seluruh 16 konfigurasi, baseline LR (`baseline_train`) memberi AUC 0,8661 dan AP 0,0426. Tabel berikut memuat selisih `hybrid_val` terhadap `lr_val_control`.

**Tabel 3a. $\Delta$AUC `hybrid_val` vs `lr_val_control`** (CI bootstrap di atas nol pada 0 dari 3 seed untuk semua sel)

| Arsitektur | $W=1$   | $W=3$   | $W=5$   | $W=10$  |
| ---------- | ------- | ------- | ------- | ------- |
| plain      | +0,0001 | +0,0016 | +0,0011 | +0,0014 |
| bottleneck | +0,0001 | +0,0005 | +0,0015 | +0,0011 |
| attn       | +0,0003 | +0,0015 | +0,0017 | +0,0008 |
| attn_pool  | +0,0000 | +0,0017 | +0,0014 | +0,0012 |

**Tabel 3b. $\Delta$AP `hybrid_val` vs `lr_val_control`**

| Arsitektur | $W=1$   | $W=3$   | $W=5$   | $W=10$  |
| ---------- | ------- | ------- | ------- | ------- |
| plain      | +0,0020 | +0,0019 | +0,0018 | +0,0019 |
| bottleneck | +0,0020 | +0,0017 | +0,0024 | +0,0020 |
| attn       | +0,0018 | +0,0018 | +0,0020 | +0,0018 |
| attn_pool  | +0,0017 | +0,0018 | +0,0018 | +0,0019 |

- Kontrol null (error diacak) menghasilkan selisih AUC antara −0,0002 dan 0,0000, jauh lebih kecil daripada gain AP di atas, sehingga gain AP bukan derau murni.
- `hybrid_val` mencapai AP 0,0431–0,0438 dan *recall*@FPR 1% 0,243–0,254, dibanding `lr_val_control` 0,0414 dan 0,224. Terhadap LR produksi (`baseline_train`) gain menyusut: pada $W=1$ attn, $\Delta$AUC 0,0000 dan $\Delta$AP +0,0006 (CI tidak di atas nol).
- `stack_val` menaikkan AUC sebesar +0,0022 sampai +0,0034 terhadap `baseline_train` (CI di atas nol pada 3 dari 3 seed untuk semua 16 konfigurasi), tetapi menurunkan AP sekitar −0,0020 sampai −0,0025.
- `hybrid_train_naive` (protokol bocor) mencapai *recall*@FPR 1% 0,256–0,271, lebih tinggi dari `hybrid_val`; selisih ini adalah efek kebocoran, bukan kemampuan model, sehingga tidak dilaporkan sebagai hasil.
- Sebagai fitur tunggal, `ae_error_only` memberi AUC dari 0,6710 (plain/bottleneck, $W=1$) hingga 0,4905 (bottleneck, $W=5$).

### 4.4 Stratifikasi menurut panjang riwayat (AUC uji)

| Bucket | n       | n fraud | LR full | LR no_history | LR history_only | GBM   | hybrid_val ($W=1$, attn) |
| ------ | ------- | ------- | ------- | ------------- | --------------- | ----- | ------------------------ |
| 1      | 76.436  | 810     | 0,847   | 0,864         | 0,500           | 0,841 | 0,8451                   |
| 2-4    | 139.439 | 239     | 0,772   | 0,720         | 0,517           | 0,846 | 0,7763                   |
| 5-9    | 110.007 | 109     | 0,759   | 0,681         | 0,556           | 0,805 | 0,7564                   |
| >=10   | 89.161  | 76      | 0,721   | 0,649         | 0,489           | 0,772 | 0,7152                   |

AUC turun monoton dengan panjang riwayat pada semua model berlabel, dan 810 dari 1.234 fraud uji berada di bucket 1. AUC `history_only` mendekati 0,5 di dalam setiap bucket, jadi AUC keseluruhan 0,7618 hampir seluruhnya berasal dari pemisahan antar bucket, bukan dari diskriminasi di dalam bucket.

### 4.5 Diagnostik "identity collapse" dan uji bottleneck ekstra-kecil (2026-11-08)

Untuk menguji apakah ceiling pada bagian 4.2 disebabkan oleh bottleneck yang masih terlalu longgar (sehingga model dapat menyalin/"collapse" ke solusi yang mengecilkan loss tanpa benar-benar membedakan pola), tiga varian bottleneck ekstra-kecil diuji pada $W=1$ dan $W=3$: dimensi laten $z \in \{2,3,4\}$ dengan hidden size diperkecil ke 16 (dari 64). Checkpoint dipilih dengan dua aturan untuk perbandingan langsung: (a) *val reconstruction loss* (protokol valid, dipakai pada bagian 4.2–4.4), dan (b) *val AUC* (protokol tidak valid, dipakai hanya sebagai diagnostik, bukan untuk pelaporan hasil).

| Arsitektur          | $W$ | Pilih checkpoint        | Epoch | Val AUC                     | Test AUC | Test AP |
| ------------------- | --- | ----------------------- | ----- | --------------------------- | -------- | ------- |
| bottleneck\_z2\_h16 | 1   | val\_loss               | 15    | 0,4930 (puncak 0,8163 @ep1) | 0,5007   | 0,0088  |
| bottleneck\_z3\_h16 | 1   | val\_loss               | 15    | 0,4357 (puncak 0,8023 @ep1) | 0,4499   | 0,0081  |
| bottleneck\_z3\_h16 | 1   | val\_auc *(diagnostik)* | 1     | 0,8023                      | 0,8070   | 0,0135  |
| bottleneck\_z4\_h16 | 3   | val\_loss               | 15    | 0,4140                      | 0,4236   | 0,0067  |

Polanya konsisten di ketiga konfigurasi: AUC validasi memuncak pada epoch 1 (hampir sebelum pelatihan berarti terjadi) lalu menurun terus hingga mendekati atau di bawah acak begitu *reconstruction loss* terus membaik — ciri khas *identity collapse*, yaitu model mulai mempelajari jalan pintas rekonstruksi generik yang menekan loss tetapi menghapus sinyal pembeda anomali. Dengan protokol valid (val\_loss, baris 1/2/4), ketiga bottleneck ekstra-kecil tidak mengalahkan arsitektur bagian 4.2 (AUC uji 0,42–0,50 vs 0,37–0,67 sebelumnya) — mengecilkan bottleneck tidak menyelesaikan masalah, bahkan sedikit memperburuk rata-rata. Satu-satunya angka tinggi (0,8070) hanya muncul bila checkpoint dipilih dari AUC validasi, yang menggunakan label untuk pemilihan model (kebocoran) pada satu seed dan satu epoch; AP-nya (0,0135) juga jauh di bawah LR full (0,0426), tidak konsisten dengan model yang benar-benar belajar memisahkan fraud. Baris ini disertakan sebagai bukti pendukung diagnosis *collapse*, bukan sebagai hasil yang diklaim.

## 5. Pembahasan

**RQ1: LSTM-AE mandiri tidak kompetitif.** Dengan AUC 0,49–0,67, semua varian LSTM-AE kalah dari LR berlabel (0,8661), gradient boosting (0,8592), dan Isolation Forest tanpa label (0,8665). Hasil ini konsisten dengan kehati-hatian bahwa menumpuk komponen tidak otomatis lebih baik (Almuteer dkk.). Kemampuan peringkat teratas lebih baik pada gradient boosting (AP 0,2938) daripada pada seluruh pendekatan berbasis rekonstruksi, yang menyiratkan interaksi non-linear antar fitur riwayat dan nominal adalah sumber sinyal yang tidak ditangkap LR maupun AE. Diagnostik bottleneck ekstra-kecil (bagian 4.5) menunjukkan ceiling ini bukan sekadar soal kapasitas bottleneck yang belum cukup kecil: memperkecilnya lebih lanjut memicu *identity collapse* (AUC validasi memuncak di epoch 1 lalu runtuh) alih-alih memperbaiki pemisahan, sehingga penyetelan hyperparameter lanjutan pada paradigma rekonstruksi ini kecil kemungkinan menutup kesenjangan terhadap baseline.

**RQ2: attention tidak terbukti membantu.** Selisih antar arsitektur pada window yang sama umumnya berada dalam satu hingga dua simpangan baku. Pada $W=5$ `attn` unggul sekitar 0,021 atas `plain`, sedangkan pada $W=1$ `plain` unggul atas `attn`, sehingga tidak ada arah yang konsisten. Di hybrid, perbedaan antar arsitektur sekitar 0,001 atau kurang untuk $\Delta$AUC, dan `attn_pool` tidak lebih baik daripada `plain`. Bottleneck murni gagal pada $W \geq 3$, kemungkinan karena satu vektor ringkasan dari window yang sebagian besar berisi padding tidak informatif. Dengan demikian, sumbangan attention terhadap deteksi tidak terdukung data ini. Analisis bobot attention (fraud vs normal) belum dilakukan sehingga aspek interpretabilitas dari RQ2 belum terjawab.

**RQ3: riwayat akun lebih penting daripada window.** Menghapus fitur riwayat menurunkan AUC LR dari 0,8661 menjadi 0,8027 dan menyebabkan AP turun separuh (0,0426 menjadi 0,0211). Efek ini hampir seluruhnya berasal dari pemisahan antar bucket (bagian 4.4), konsisten dengan *lift* 8,34 pada bucket 1. Perubahan $W$ tidak menggeser hasil hybrid secara konsisten; $\Delta$AUC terbesar muncul pada $W=3$–$5$ tetapi tetap tidak signifikan. AUC menurun pada riwayat panjang; kemungkinan penyebabnya fraud yang langka di bucket tersebut (76 dari 1.234 pada bucket >=10) sehingga estimasinya kurang stabil, tetapi ini belum diuji.

**Nilai tambah error AE kecil.** Fitur error memberi gain AP yang konsisten (sekitar 4–6% relatif terhadap `lr_val_control`) dan menaikkan *recall* pada FPR 1%, tetapi tidak menaikkan AUC secara signifikan dan hampir hilang terhadap LR produksi. `stack_val` menunjukkan *trade-off*: AUC naik signifikan, AP turun, artinya error AE memperbaiki peringkat menyeluruh sambil mengurangi ketajaman di puncak peringkat. Sebuah probe awal ($W=1$, bottleneck, satu seed, pada checkpoint dan baseline lama) menemukan error AE 44–87 kali lebih tinggi pada fraud yang terlewat LR, sehingga sinyalnya tampak komplementer tetapi terkonsentrasi pada sebagian kecil kasus. Ukuran gain yang kecil dibandingkan loncatan gradient boosting menunjukkan bahwa menaikkan kapasitas model tabular lebih menguntungkan daripada menambahkan fitur rekonstruksi. Normalisasi error per bucket riwayat kini menggunakan statistik train-normal (leak-free) namun tetap tidak memberikan peningkatan signifikan pada AUC. Interval kepercayaan bootstrap untuk recall pada FPR 1% dan 0,1% telah ditambahkan untuk memberikan ukuran ketidakpastian yang lebih lengkap.

## 6. Keterbatasan dan Kesimpulan

**Keterbatasan.** (1) Dataset sintetis dengan label tingkat transaksi dan rentang sekitar 31 hari; hasil tidak dapat dibaca sebagai bukti identifikasi pelaku pencucian uang, dan *concept drift* jangka panjang tidak diuji. (2) Hanya 3 seed (rencana awal 5); CI antar seed kasar. (3) Hanya hybrid berbasis Logistic Regression yang dievaluasi; gradient boosting dipakai sebagai rujukan, belum dengan fitur error AE. (4) Sebagian besar fraud (810 dari 1.234 di uji) berada pada akun dengan riwayat satu transaksi, sehingga window panjang jarang memberi konteks. (5) F1/MCC pada threshold, strategi threshold (p99 normal error vs labeled validation), sensitivitas prevalensi, ablation fitur selain `no_history` (fitur waktu, fitur relasional), analisis per segmen (transaction_type, period), dan visualisasi bobot attention belum dikerjakan. (6) Dense AE latent sweep (--latent-dim 2,3,4) belum dievaluasi. (7) Analisis bobot attention untuk interpretabilitas RQ D2 belum dilakukan. (8) Hybrid dengan non-linear learner (GBM) belum dievaluasi. (9) `account_code` (id factorize penerima) dikeluarkan dari fitur LR/GBM karena bukan sinyal perilaku dan berisiko membawa informasi urutan kemunculan; keputusan ini konsisten diterapkan di baseline, hybrid, dan unsupervised baseline. (10) AUC `dense_ae_seq5` berada di bawah acak (0,3590 ± 0,0102) dan belum membaik setelah fitur `amount`/`history_length` di-log1p sebelum standardisasi (sebelumnya 0,4097) — penyebab pastinya belum ditemukan, sehingga angka ini dilaporkan sebagai baseline yang diketahui bermasalah, bukan bukti bahwa dense autoencoder secara inheren anti-korelasi dengan fraud. (11) Proses pelatihan di Windows pernah berakhir dengan kode keluar 3221226505 setelah report tersimpan; belum diselidiki. (12) Diagnostik bottleneck ekstra-kecil (bagian 4.5) baru dijalankan pada satu seed (42) tanpa interval kepercayaan; pola *identity collapse*-nya konsisten secara kualitatif di tiga konfigurasi tetapi belum dikonfirmasi lewat ulangan multi-seed sebelum dianggap final.

**Kesimpulan.** Pada PaySim yang berpusat pada akun penerima, LSTM-Autoencoder dengan attention tidak kompetitif sebagai detektor mandiri dan tidak mengungguli Isolation Forest maupun Logistic Regression. Sebagai fitur tambahan, error rekonstruksi hanya memberi gain AP kecil dan tidak signifikan di AUC, independen dari varian attention. Sinyal dominan berasal dari panjang riwayat akun penerima dan interaksi non-linear antar fitur. Untuk data dengan riwayat akun tipis, pendekatan berbasis rekonstruksi bukan sumber peningkatan yang menjanjikan dibanding memperkaya fitur dan kapasitas model tabular.

## Daftar Pustaka

- Achituve dkk. (2019). Interpretable online banking fraud detection. [doi.org/10.1109/MLSP.2019.8918896](https://doi.org/10.1109/MLSP.2019.8918896)
- Almuteer dkk. Perbandingan CNN, AE, LSTM, dan AE+LSTM. [igi.indrastra.com](https://igi.indrastra.com/items/show/2167)
- Benchaji dkk. (2021). Attention LSTM untuk klasifikasi fraud. [doi.org/10.1186/s40537-021-00541-8](https://doi.org/10.1186/s40537-021-00541-8)
- Ding dkk. (2024). Deteksi fraud sebagai anomaly detection. [peerj.com](https://peerj.com/articles/cs-2323/)
- Kolawole (2023). Autoencoder-LSTM dengan SMOTE. [norma.ncirl.ie](http://norma.ncirl.ie/7550/)
- Umaru dkk. (2025). FUDMA Journal of Sciences. [fjs.fudutsinma.edu.ng](https://fjs.fudutsinma.edu.ng/index.php/fjs/article/view/3072)
- Verma dan Dhar (2024). [arxiv.org/abs/2409.13406](https://arxiv.org/abs/2409.13406)
- PaySim. [kaggle.com/datasets/ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1)