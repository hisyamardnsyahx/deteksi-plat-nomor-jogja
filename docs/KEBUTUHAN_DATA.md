# Kebutuhan data — langkah B

## B1. Yang harus dipenuhi data training & testing

Data Roboflow (614 gambar) menjawab “di mana platnya”, bukan
“plat mana yang Jogja”. Jadi data baru harus menjawab pertanyaan kedua.

### Jumlah (target, bukan angka sakti)

- Total baru: 600–1.000 gambar berlabel Jogja/non-Jogja.
- Per kelas: 300–500 Jogja (AB) dan 300–500 non-Jogja.
- Test khusus Jogja: min. 100 gambar, 50 Jogja / 50 non-Jogja,
  kendaraan, hari, dan lokasi berbeda dari train.
- Pilot minimum yang masih bisa dilaporkan: 100/kelas untuk train
  + 25/kelas untuk test — tapi ekspektasikan akurasi huruf masih goyah,
  terutama plat jauh/blur, dan tulis itu sebagai keterbatasan.

### Ukuran & kualitas

- Simpan foto asli HP (jangan diperkecil dulu di HP). Training yang
  resize: Ultralytics letterbox, bukan stretch seperti preprocessing
  Roboflow 720×720 (stretch mengubah rasio plat dan bentuk huruf).
- Plat harus cukup besar untuk dibaca manusia di foto: target lebar
  plat di foto minimal ~120 px sebelum crop; kalau manusia saja tidak
  bisa membaca prefixnya, jangan dipaksakan jadi data training —
  masukkan ke kategori “tidak terbaca” dan laporkan jumlahnya.
- Format: JPG, sRGB. Satu kendaraan boleh muncul beberapa kali hanya
  kalau sudut/jarak/cahaya benar-benar berbeda, dan semua fotonya
  harus satu split (group split) supaya tidak bocor.

### Komposisi (matriks yang harus diisi, bukan sekadar total)

- Waktu: siang terang, mendung, sore, malam (lampu).
- Posisi: depan & belakang, lurus & miring (±30°), dekat & jauh.
- Kendaraan: motor & mobil.
- Plat: hitam lama & putih baru, bersih & kotor/blur ringan.
- Latar: jalan, parkir, bengkel/cuci motor — jangan semuanya dari
  satu parkiran yang sama.
- Non-Jogja: sebar kode wilayahnya (B, D, L, AA, AD, AB? tidak —
  AB itu Jogja; contoh non: B, D, L, AA, AD, E, G, H, K, N, S, W, dll.),
  jangan hanya satu kode (mis. hanya B) karena model akan belajar
  “B vs AB”, bukan “non-Jogja vs AB”.

## B2. Ambil dari dunia nyata

- Foto dengan HP sendiri, kendaraan sendiri / dengan izin pemilik,
  atau kendaraan di area publik yang platnya memang terpampang.
- Jangan mengambil foto plat orang dari internet tanpa lisensi dan
  jangan mempublikasikan daftar lengkap nomor plat di laporan —
  di laporan cukup tampilkan prefix (AB / non-AB) dan contoh crop
  yang sudah diburamkan sebagian nomornya.
- Screenshot layer (mis. dari video CCTV/dashcam) boleh, tapi catat
  sumbernya sebagai split terpisah — kualitasnya biasanya lebih rendah.

## B3. Penyesuaian data

- Crop: TIDAK memotong plat sebelum anotasi. Anotasi di foto utuh,
  kotak ketat di sekeliling plat saja.
- Rotate/shift: rotasi kecil (±10°), geser, brightness/contrast boleh.
- DILARANG untuk tugas ini: flip horizontal/vertikal (membalik huruf),
  crop yang memotong huruf prefix, dan stretch yang mengubah rasio.
- Augmentasi Ultralytics saat training tetap jalan (mosaic dll.),
  tapi dengan `fliplr=0.0` (lihat README).

## B4. Anotasi

- Tool: Roboflow Annotate atau LabelImg, ekspor YOLO txt.
- Satu baris per plat: `<class> <x_center> <y_center> <w> <h>`
  (semua ternormalisasi 0–1).
- Kelas detektor 1 kelas: `0` = plat.
- Kelas 2 kelas: `0` = plat-jogja, `1` = plat-non-jogja.
- WAJIB berdampingan: CSV audit `image, plate_text_dibaca_manusia,
  prefix, label_jogja, keterangan_kualitas` — ini ground truth untuk
  mengukur OCR, bukan sekadar label kotak.
- Dua orang membaca (double-check) untuk minimal 20% data, terutama
  plat yang hurufnya samar. Selisih baca = data “tidak yakin”.

## B5. Keputusan setelah inference pretrained

Catat di `runs/` dalam format: jumlah test, deteksi benar, plat
terlewat, salah kotak, OCR benar/salah, confusion matrix.
Baru dari tabel itu diputuskan: perbaiki data, fine-tune detektor,
atau perbaiki OCR — jangan memutuskan dari kesan “kayaknya bagus”.
