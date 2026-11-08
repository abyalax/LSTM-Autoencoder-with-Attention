
TODO for pre-submit paper

P0 — Validitas metodologi
[ ] Audit fitur historis — Pastikan history_length, history_bucket, dan fitur turunan hanya menggunakan informasi masa lalu.
[ ] Audit sequence construction — Pastikan sorting, timestamp tie-breaking, windowing, padding, dan masking deterministik.
[ ] Audit split dan leakage — Verifikasi pemisahan account-level, preprocessing, dan independensi train/validation/test.
[ ] Audit hybrid metrics — Tambahkan null delta AP dan paired bootstrap CI untuk delta AP.

P1 — Penguatan eksperimen
[ ] Lengkapi metrik LSTM-AE — Laporkan AP, Recall@FPR 1%, dan Recall@FPR 0,1% untuk seluruh konfigurasi.
[ ] Perkuat bootstrap — Gunakan cluster bootstrap berbasis penerima dan minimal 2.000 resamples.
[ ] Lengkapi parameter model — Dokumentasikan optimizer, learning rate, batch size, patience, decoder, latent dimension, dan parameter count.
[ ] Perkuat analisis bucket — Tambahkan AP, fraud rate, confidence interval, dan analisis ketidakpastian per bucket.
[ ] Validasi baseline GBM — Periksa implementasi AP dan konsistensi evaluasi terhadap LR.
[ ] Revisi diagnosis collapse — Ubah klaim identity collapse menjadi hipotesis atau tambahkan bukti representasi laten.

P1 — Revisi narasi ilmiah
[ ] Perbaiki research gap — Gunakan tiga gap yang langsung berhubungan dengan eksperimen.
[ ] Perbaiki RQ2 — Pisahkan evaluasi performa attention dari interpretabilitas atau hapus aspek interpretabilitas.
[ ] Perkuat pembahasan — Jelaskan mekanisme dan kemungkinan penyebab hasil, bukan sekadar mengulang tabel.
[ ] Persempit kesimpulan — Batasi generalisasi pada dataset dan konfigurasi yang diuji.
[ ] Revisi judul dan abstrak — Tonjolkan evaluasi reconstruction error, bukan klaim peningkatan attention.

P2 — Editorial
[ ] Lengkapi referensi — Nama penulis, judul, tahun, jurnal, volume, issue, halaman, DOI/URL.
[ ] Rapikan keterbatasan — Hapus duplikasi attention dan perbaiki typo RQ D2.
[ ] Tambahkan visualisasi — Diagram pipeline, arsitektur model, dan grafik performa per bucket.