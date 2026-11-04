# Kebutuhan data: langkah B

Data Roboflow yang dipakai sejauh ini menjawab di mana platnya,
bukan plat mana yang Jogja. Data baru harus menjawab pertanyaan
kedua itu, dan dokumen ini menuliskan syaratnya supaya pengumpulan
datanya tidak asal banyak.
 
## Jumlah

Targetnya 600 sampai 1.000 gambar baru yang sudah berlabel Jogja
atau non-Jogja, dengan 300 sampai 500 gambar per kelas. Di samping
itu, siapkan test khusus minimal 100 gambar, 50 Jogja dan 50
non-Jogja, yang kendaraan, hari, dan lokasinya berbeda dari train.
Angka-angka ini target kerja, bukan jaminan akurasi.

Bila waktu atau akses terbatas, pilot minimum yang masih bisa
dilaporkan adalah 100 gambar per kelas untuk train dan 25 per
kelas untuk test. Dengan jumlah segitu, akurasi membaca huruf
kemungkinan masih goyah, terutama untuk plat yang jauh atau blur,
dan keterbatasan itu harus ditulis di laporan, bukan disembunyikan.

## Ukuran dan kualitas

Simpan foto asli dari HP, jangan diperkecil dulu di HP. Resize
saat training dilakukan dengan letterbox oleh Ultralytics, bukan
dengan stretch seperti preprocessing Roboflow ke 720x720, karena
stretch mengubah rasio plat dan bentuk huruf.

Plat harus cukup besar untuk dibaca manusia di foto. Patokannya,
lebar plat minimal sekitar 120 piksel sebelum crop. Bila manusia
saja tidak bisa membaca prefixnya, jangan dipaksakan menjadi data
training. Masukkan foto itu ke kategori tidak terbaca dan laporkan
jumlahnya.

Formatnya JPG, sRGB. Satu kendaraan boleh muncul beberapa kali
hanya bila sudut, jarak, atau cahayanya benar-benar berbeda, dan
semua fotonya harus berada dalam satu split yang sama. Ini yang
disebut group split: foto kendaraan yang sama tidak boleh bocor
ke train sekaligus test.

## Komposisi

Total saja tidak cukup; variasinya harus terisi. Waktu pengambilan
perlu mencakup siang terang, mendung, sore, dan malam dengan lampu.
Posisi plat perlu mencakup depan dan belakang, lurus dan miring
sekitar 30 derajat, dekat dan jauh. Kendaraannya motor dan mobil.
Platnya plat hitam lama dan putih baru, yang bersih maupun yang
kotor atau blur ringan. Latarnya jangan dari satu parkiran yang
sama saja; jalan, parkir, dan bengkel atau tempat cuci motor
semuanya boleh, asal beragam.

Untuk kelas non-Jogja, sebar kode wilayahnya. Jangan hanya memakai
satu kode, misalnya hanya B, karena model kemudian hanya belajar
membedakan B dengan AB, bukan non-Jogja dengan AB. Contoh kode
yang bisa dipakai: B, D, L, AA, AD, E, G, H, K, N, S, W, dan
lain-lain. AB sendiri tentu bukan non-Jogja.

## Pengambilan dari dunia nyata

Foto diambil dengan HP sendiri, atas kendaraan sendiri atau dengan
izin pemiliknya, atau atas kendaraan di area publik yang platnya
memang terpampang. Jangan mengambil foto plat orang dari internet
tanpa lisensi. Di laporan, jangan mempublikasikan daftar lengkap
nomor plat. Cukup tampilkan prefix, AB atau non-AB, dan contoh
crop yang nomornya sudah diburamkan sebagian.

Screenshot dari layer lain, misalnya video CCTV atau dashcam,
boleh dipakai, tetapi catat sumbernya sebagai split terpisah.
Kualitasnya biasanya lebih rendah, dan perbedaan itu perlu
terlihat di analisis, bukan tercampur diam-diam dengan foto HP.

## Penyesuaian data

Anotasi dilakukan pada foto utuh, dengan kotak yang ketat
mengelilingi plat saja. Jangan memotong plat sebelum anotasi.
Rotasi kecil sekitar 10 derajat, geser, dan penyesuaian brightness
atau contrast masih diperbolehkan. Tiga hal dilarang untuk tugas
ini: flip horizontal atau vertikal karena membalik huruf, crop
yang memotong huruf prefix, dan stretch yang mengubah rasio.
Augmentasi Ultralytics saat training tetap berjalan, termasuk
mosaic, tetapi dengan `fliplr=0.0`, sesuai README.

## Anotasi

Alatnya bisa Roboflow Annotate atau LabelImg, dengan ekspor format
YOLO txt. Satu baris per plat: kelas, x_center, y_center, lebar,
dan tinggi, semuanya ternormalisasi antara 0 dan 1. Untuk detektor
satu kelas, kelasnya 0 untuk plat. Untuk dua kelas, 0 untuk
plat-jogja dan 1 untuk plat-non-jogja.

Yang wajib berdampingan dengan file txt itu adalah CSV audit
dengan kolom: nama gambar, teks plat hasil baca manusia, prefix,
label Jogja, dan keterangan kualitas. CSV ini adalah ground truth
untuk mengukur OCR, bukan sekadar label kotak. Untuk minimal 20
persen data, terutama plat yang hurufnya samar, mintalah dua orang
membaca secara terpisah. Bila hasil bacanya berbeda, data itu
masuk kategori tidak yakin.

## Keputusan setelah inference

Setelah inference dengan model yang sudah ada, catat hasilnya di
`runs/` dalam bentuk tabel: jumlah test, deteksi benar, plat
terlewat, salah kotak, OCR benar dan salah, serta confusion
matrix. Baru dari tabel itu diputuskan apakah yang perlu diperbaiki
adalah datanya, detektornya lewat fine-tuning, atau OCR-nya.
Keputusan itu jangan diambil dari kesan bahwa hasilnya kayaknya
sudah bagus.
