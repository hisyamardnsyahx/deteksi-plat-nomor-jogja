# Menjalankan training di Google Colab (karena spesifikasi VM kurang)

VM lokal: CPU 4-core, RAM 7,7 GB, tanpa GPU. Pilot lokal terbukti lambat:
~3 detik/batch, 62 batch/epoch ≈ 3 menit/epoch di imgsz 640 — 30 epoch
≈ 90 menit, dan training penuh 100 epoch di imgsz 960 akan jauh lebih
lama lagi. Karena itu training penuh dipindah ke Colab, sesuai instruksi
Syam 4 Nov 2026. Pilot lokal dihentikan di epoch 4/30 (checkpoint pilot
tetap ada di `runs/detect/`, tidak dipakai untuk hasil utama).

## File notebook

`colab_yolo11_plat_jogja.ipynb` (13 sel, sudah divalidasi JSON).

Isi notebook, berurutan:
1. Cek GPU (wajib T4, kalau tidak terdeteksi notebook berhenti dengan pesan jelas).
2. Instal `ultralytics` + `gdown`.
3. Unduh ZIP dataset dari Drive Syam (file ID sudah tertulis di notebook) + ekstrak.
4. Audit Jogja/non-Jogja dari nama file + 1 koreksi manual (file `K4_1938WA` = AB).
5. Relabel 1 kelas → 2 kelas, split asli dipertahankan, kotak degeneratif dibuang.
6. Baseline pretrained COCO di test (ekspektasi: 0 deteksi plat).
7. **Training penuh 2 kelas:** `yolo11n.pt`, epochs 100, imgsz 960, batch 16,
   patience 50, seed 42, `fliplr=0.0` (jangan diubah — flip membalik huruf).
8. Evaluasi standar valid + test (mAP50, mAP50-95).
9. Confusion matrix test (benar Jogja / benar non-Jogja / salah kelas /
   terlewat / false positive).
10. Unduh hasil (`best.pt`, `results.csv`, `eval_test.csv`, `audit_jogja.csv`).

## Langkah menjalankan (dari HP pun bisa, tapi lebih nyaman dari laptop)

1. Buka [Google Colab](https://colab.research.google.com), pilih
   **File → Upload notebook**, pilih file `.ipynb` di atas.
2. **Runtime → Change runtime type → Hardware accelerator: T4 GPU** → Save.
   Kalau tertulis “None” atau “CPU”, training akan selambat VM lokal — jangan lanjutkan.
3. **Runtime → Run all.** Biarkan berjalan; jangan tutup tab sampai sel 10 selesai.
   Perkiraan kasar di T4: puluhan menit untuk 100 epoch (bukan patokan pasti —
   tergantung antrean GPU dan ukuran batch).
4. Kalau muncul error `CUDA out of memory` di sel training: ubah `batch=16`
   menjadi `batch=8` di sel 7, lalu jalankan ulang sel 7 saja, dan catat
   perubahan itu untuk laporan.
5. Setelah sel 10, unduhan `hasil_jogja.zip` + CSV akan mulai otomatis.
   Simpan `best.pt` ke Drive kalau mau dipakai lagi.

## Yang dikirim kembali untuk laporan

- `results.csv` (mAP valid),
- angka test sel 8 + ringkasan sel 9,
- `best.pt` (opsional, untuk inference lokal),
- catatan perubahan konfigurasi apa pun (batch, early stopping, dll.).

## Catatan kejujuran yang tetap berlaku di Colab

Label 2 kelas berasal dari nama file + 1 verifikasi visual, bukan
pembacaan manual 614 foto. Sebelum klaim final di laporan, spot-check
visual acak 10% (daftar di `runs/spotcheck_sample.csv`) tetap harus
dilakukan dan hasilnya dicatat.
