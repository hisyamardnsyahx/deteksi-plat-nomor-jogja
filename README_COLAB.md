# Menjalankan training di Google Colab

Training penuh dipindah ke Colab karena mesin lokal tidak punya GPU.
Mesin lokal itu CPU 4-core dengan RAM 7,7 GB. Pilot lokalnya berjalan
sekitar 3 menit per epoch pada ukuran gambar 640, jadi 30 epoch saja
sudah sekitar 90 menit, dan 100 epoch pada ukuran 960 akan jauh lebih
lama. Pilot itu dihentikan di epoch 4 dari 30. Checkpoint-nya masih
ada di `runs/detect/`, tetapi tidak dipakai untuk hasil utama.

## Isi notebook

File `colab_yolo11_plat_jogja.ipynb` berisi 20 sel: 10 sel markdown
dan 10 sel kode. Alurnya, berurutan, adalah cek GPU, instalasi,
unduh dataset, audit label, relabel, baseline, training, evaluasi,
confusion, lalu prediksi dan unduhan hasil. Bagian trainingnya
memakai `yolo11n.pt`, 100 epoch, ukuran gambar 960, batch 16,
patience 50, seed 42, dan tanpa flip. Bagian evaluasinya memakai
`model.val()` standar untuk validasi dan test, ditambah confusion
matrix pada test dengan IoU 0,5 dan confidence 0,25.

## Langkah menjalankan

Buka notebook di Google Colab, dari repo ini atau dengan mengunggah
file `.ipynb` lewat menu File. Sebelum menjalankan apa pun, atur
runtime-nya: Runtime, Change runtime type, Hardware accelerator
pilih T4 GPU, lalu simpan. Bila runtime tertulis None atau CPU,
jangan dilanjutkan, karena training akan selambat mesin lokal.

Setelah itu pilih Runtime, Run all, dan biarkan berjalan sampai sel
terakhir selesai. Jangan tutup tab di tengah jalan. Pada GPU T4,
100 epoch pada run referensi memakan waktu sekitar 24 menit, tetapi
angka itu bukan patokan pasti; antrean GPU dan ukuran batch bisa
mengubahnya.

Bila muncul error kehabisan memori CUDA pada sel training, ubah
`batch=16` menjadi `batch=8` di sel itu saja, jalankan ulang sel
tersebut, dan catat perubahannya. Hasilnya tidak lagi identik
dengan run referensi, dan perbedaan itu harus tertulis di laporan.

Setelah sel terakhir selesai, unduhan akan mulai otomatis:
`hasil_jogja.zip` yang berisi `best.pt`, `results.csv`, dan grafik
training, ditambah `eval_test.csv` dan `audit_jogja.csv`. Simpan
ketiganya. Bila bobotnya mau dipakai lagi nanti, salin `best.pt`
ke Drive sebelum runtime Colab berakhir.

## Yang dikirim untuk laporan

Yang perlu dikirim kembali adalah `results.csv` dari folder
training, angka test dari sel evaluasi, ringkasan confusion, dan
`best.pt` bila inference lokal diperlukan. Sertakan juga catatan
atas perubahan konfigurasi apa pun, misalnya batch yang diturunkan
atau training yang berhenti lebih awal karena early stopping.

## Catatan tentang label

Satu hal yang tetap berlaku di Colab maupun di lokal: label dua
kelas berasal dari nama file ditambah satu koreksi manual yang
diverifikasi visual, bukan dari pembacaan manual atas 614 foto.
Sebelum angka-angka ini diklaim final di laporan, spot-check
visual acak 10 persen, yang daftarnya ada di
`runs/spotcheck_sample.csv`, tetap harus dilakukan dan hasilnya
dicatat.
