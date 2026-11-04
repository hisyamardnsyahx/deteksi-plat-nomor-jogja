# Relabel Roboflow 1 kelas -> Jogja 2 kelas

Masalah: label YOLO bawaan Roboflow hanya menyimpan id kelas + kotak.
Ia tidak menyimpan teks plat. Jadi **tidak ada cara otomatis yang sah**
untuk mengubah 614 label itu menjadi Jogja/non-Jogja tanpa membaca
tiap platnya.

Prosedur yang benar:

1. Jalankan `scripts/audit_dataset.py` untuk memastikan struktur
   train/valid/test dan menghitung kotak per split.
2. Untuk tiap gambar, baca prefix plat secara manual (atau OCR +
   verifikasi manual 100% — OCR saja tidak cukup jadi ground truth).
3. Isi CSV: `image, prefix, label` dengan label `plat-jogja` kalau
   prefix == AB, `plat-non-jogja` kalau bukan, `unreadable` kalau
   tidak terbaca.
4. Jalankan `scripts/relabel_to_jogja.py --csv audit.csv
   --src data/roboflow --dst data/jogja_2cls`.
   Skrip ini menyalin gambar + menulis ulang id kelas di txt
   (0/1), membuang `unreadable`, dan MEMBUAT split baru yang
   stratified per kelas — split bawaan Roboflow tidak bisa dipakai
   mentah-mentah lagi karena proporsi AB/non-AB di tiap split lama
   tidak diketahui sebelum audit.
5. Simpan CSV audit sebagai lampiran laporan. Tanpa CSV ini,
   klaim akurasi Jogja tidak bisa diverifikasi.

Catatan: kalau ternyata mayoritas data Roboflow itu AB (atau
sebaliknya), JANGAN menyeimbangkan dengan menduplikasi gambar yang
sama berkali-kali. Tambah data baru dari kelas minoritas (langkah B),
atau laporkan ketidakseimbangan + pakai metrik per kelas
(precision/recall per kelas, bukan hanya akurasi total).
