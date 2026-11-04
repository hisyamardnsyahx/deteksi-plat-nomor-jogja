# Relabel Roboflow satu kelas menjadi dua kelas Jogja

## Masalahnya

Anotasi YOLO bawaan Roboflow hanya menyimpan id kelas dan koordinat
kotak. Ia tidak menyimpan teks plat, sehingga tidak ada cara yang
sah untuk mengubah 614 label itu menjadi Jogja dan non-Jogja tanpa
membaca tiap platnya.

Prosedur yang ideal memang begitu: baca prefix tiap plat secara
manual, atau pakai OCR yang kemudian diverifikasi manual seratus
persen, karena OCR saja tidak cukup menjadi ground truth. Hasil
bacaan itu ditulis ke CSV berisi nama gambar, prefix, dan label:
`plat-jogja` bila prefixnya AB, `plat-non-jogja` bila bukan, dan
`unreadable` bila tidak terbaca. CSV itu kemudian menjadi lampiran
laporan, karena tanpa CSV tersebut klaim akurasi Jogja tidak bisa
diverifikasi.

## Yang benar-benar dilakukan di proyek ini

Proyek ini tidak mengikuti prosedur ideal itu sampai tuntas, dan
perbedaan itu perlu ditulis apa adanya. Nama file Roboflow ternyata
mengkodekan teks plat, misalnya `K1_AB2430VL_...jpg` berarti plat
AB 2430 VL, sehingga label dibuat dari pola nama file tersebut,
bukan dari pembacaan manual atas tiap foto. Alurnya: jalankan
`scripts/audit_dataset.py` untuk memastikan struktur train, valid,
dan test serta menghitung kotak per split, lalu
`scripts/generate_audit_csv.py` untuk menulis CSV audit dari nama
file, lalu `scripts/relabel_to_jogja.py` untuk menyalin gambar dan
menulis ulang id kelas pada file txt menjadi 0 untuk Jogja dan 1
untuk non-Jogja.

Satu pengecualian ditangani manual, bukan ditebak. File test
`K4_1938WA...jpg` tidak membawa prefix di namanya, padahal fotonya
AB 1938 WA, sehingga labelnya dikoreksi dari pembacaan foto
langsung. Satu kotak train yang degeneratif, sekitar 0,002x0,002,
dibuang sebagai noise anotasi. Split asli dipertahankan, bukan
diacak ulang dan bukan dibuat ulang secara stratified, karena
pemeriksaan menunjukkan tidak ada token plat yang sama muncul di
dua split berbeda dan tidak ada duplikat token di dalam satu split.

## Konsekuensinya

Karena label berasal dari nama file, label itu sekuat nama file
itu sendiri. Pemeriksaan konsistensi label terhadap nama file
untuk seluruh 614 gambar menghasilkan nol ketidakcocokan selain
file K4 tersebut, tetapi pemeriksaan itu bersifat sirkular dan
hanya membuktikan konsistensi, bukan kebenaran foto. Karena itu
spot-check visual acak sebesar 10 persen, yaitu 61 gambar yang
daftarnya ada di `runs/spotcheck_sample.csv`, tetap wajib sebelum
angka-angka ini diklaim final. Bila spot-check menemukan label
yang salah, perbaiki CSV auditnya dan jalankan ulang relabel.

Satu catatan lagi untuk data berikutnya. Bila ternyata mayoritas
data adalah AB, atau sebaliknya, jangan menyeimbangkan dengan
menduplikasi gambar yang sama berkali-kali. Tambah data baru dari
kelas minoritas, sesuai langkah B, atau laporkan
ketidakseimbangannya dan pakai metrik per kelas, precision dan
recall per kelas, bukan hanya akurasi total.
