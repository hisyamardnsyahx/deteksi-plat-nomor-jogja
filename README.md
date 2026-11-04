# Deteksi Plat Nomor: Jogja vs Non-Jogja dengan YOLOv11

Proyek ini menjawab satu pertanyaan dari foto kendaraan: apakah platnya
berasal dari Yogyakarta atau bukan. Patokannya sederhana. Plat dengan
kode wilayah AB, yang mencakup Kota Yogyakarta, Sleman, Bantul, Kulon
Progo, dan Gunungkidul, dihitung sebagai Jogja. Kode wilayah lain
dihitung sebagai non-Jogja.

Soalnya tidak sesederhana menemukan kotak plat. Bentuk plat AB dan
non-AB sama persis, jadi yang membedakan hanyalah huruf di awal plat.
Model di sini harus belajar pola huruf itu dari piksel, atau membacanya
lewat OCR. Keputusan-keputusan di proyek ini, dari ukuran gambar sampai
augmentasi yang dimatikan, mengikuti konsekuensi dari fakta itu.

## Hasil

Model yang dilaporkan adalah YOLOv11n yang di-fine-tune dari
`yolo11n.pt` (pretrained COCO) menjadi detektor dua kelas,
`plat-jogja` dan `plat-non-jogja`. Training berjalan di Google Colab
dengan GPU Tesla T4: 100 epoch selesai dalam sekitar 24 menit, tanpa
early stopping dan tanpa kehabisan memori GPU.

| Split | Gambar | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Validasi | 62 | 0,985 | 0,978 | 0,995 | 0,787 |
| Test | 61 | 0,997 | 1,000 | 0,995 | 0,800 |

Pada validasi, kelas Jogja (30 instance) mencapai precision 0,97 dan
recall 1,000, sedangkan kelas non-Jogja (32 instance) mencapai
precision 1,000 dan recall 0,956, keduanya dengan mAP50 0,995.
Pada test, dengan IoU minimal 0,5 dan confidence 0,25, hasilnya 33
Jogja benar, 28 non-Jogja benar, tidak ada yang salah kelas, tidak
ada yang terlewat, dan ada 5 false positive. False positive itu
berupa kotak tambahan, bukan plat yang salah diklasifikasikan, jadi
seluruh 61 plat pada test ditemukan dan diklasifikasikan dengan benar.

Sebagai pembanding, model COCO yang belum di-fine-tune tidak menemukan
satu pun plat pada 61 gambar test. Ia menemukan 284 objek lain,
sebagian besar motor (175) dan mobil (90), sisanya orang, truk, pot
tanaman, dan ransel. Hasil itu sudah bisa diduga, karena COCO memang
tidak punya kelas plat nomor, dan ia menegaskan satu hal: tanpa
fine-tuning, tidak ada klaim deteksi plat yang bisa dibuat.

## Metode

Ada dua jalur dalam proyek ini, dan keduanya jangan dicampur.

Jalur pertama adalah yang dilatih dan dievaluasi, yaitu detektor dua
kelas langsung di atas. Model ini mengeluarkan kotak, label, dan
confidence, tetapi tidak mengeluarkan teks plat seperti "AB 2430 VL".
Ukuran gambar 960 dipilih supaya huruf prefix masih cukup terbaca
oleh model pada resolusi itu.

Jalur kedua adalah pipeline dua tahap: deteksi plat satu kelas, crop
plat dengan padding 5 sampai 10 persen, lalu OCR membaca prefixnya.
AB berarti Jogja, kode lain berarti non-Jogja, dan bila OCR tidak
yakin, hasilnya dicatat sebagai tidak yakin, bukan dipaksakan.
Jalur ini lebih mudah diaudit huruf per huruf, dan secara konsep
lebih meyakinkan untuk pertanyaan AB versus non-AB. Tapi perlu
dicatat dengan jelas: jalur ini belum dijalankan dan belum
dievaluasi di proyek ini. Kerangkanya ada di
`scripts/detect_classify.py`. Angka yang dilaporkan di atas seluruhnya
berasal dari jalur pertama.

Konfigurasi trainingnya: model awal `yolo11n.pt`, 100 epoch,
patience 50, seed 42, ukuran gambar 960, batch 16, workers 2,
optimizer SGD bawaan Ultralytics dengan learning rate awal 0,01,
momentum 0,937, weight decay 0,0005, dan warmup 3 epoch. Augmentasi
mosaic dipakai penuh dan dimatikan 10 epoch terakhir
(`close_mosaic=10`). Satu hal yang tidak boleh diubah sembarangan
adalah flip: `fliplr` dan `flipud` sama-sama nol. Flip horizontal
akan mencerminkan huruf plat, sehingga AB yang dipelajari model
bukan lagi AB yang ada di jalan. Evaluasi headline memakai
`model.val()` standar Ultralytics; confusion dihitung pada IoU 0,5
dan confidence 0,25 seperti tertulis di atas.

## Dataset dan pelabelan

Data berasal dari Roboflow Universe, "Deteksi Plat Nomor Kendaraan 2"
milik Universitas Teknologi Yogyakarta, versi 1, 16 Desember 2023,
dengan lisensi CC BY 4.0. Tautan halamannya ada di bawah, pada bagian
menjalankan ulang. Preprocessing dari sumbernya adalah Auto-Orient
dan resize stretch ke 720x720, tanpa augmentasi.

Dataset ini berisi 614 gambar: 491 train, 62 validasi, dan 61 test.
Anotasi aslinya hanya punya satu kelas, `Plat-Nomor`, dengan 492
kotak pada train, 62 pada validasi, dan 61 pada test. Gambarnya
berukuran 720x720, dan kotak platnya rata-rata selebar 0,33 dan
setinggi 0,11 dari ukuran gambar. Platnya relatif besar di dalam
frame, dan itu membantu, baik untuk deteksi maupun untuk membaca
huruf.

Karena label aslinya tidak menyimpan informasi wilayah, label Jogja
dan non-Jogja dibuat lewat relabel. Nama file Roboflow ternyata
mengkodekan teks plat, misalnya `K1_AB2430VL_...jpg` berarti plat
AB 2430 VL, dan relabel memanfaatkan pola itu. Split asli
dipertahankan, bukan diacak ulang, karena pemeriksaan menunjukkan
tidak ada token plat yang sama muncul di dua split berbeda, dan
tidak ada duplikat token di dalam satu split. Satu kotak train
yang degeneratif, sekitar 0,002x0,002, dibuang sebagai noise anotasi,
sehingga train bersih berisi 491 kotak. Hasil relabelnya:

| Split | Jogja (AB) | Non-Jogja | Total |
|---|---:|---:|---:|
| Train | 266 | 225 | 491 |
| Validasi | 30 | 32 | 62 |
| Test | 33 | 28 | 61 |
| Total | 329 | 285 | 614 |

Bagian ini perlu dibaca dengan hati-hati sebelum angkanya dikutip.
Label dua kelas ini diturunkan dari nama file, bukan dari pembacaan
manual atas 614 foto. Satu pengecualian sudah dikoreksi manual: file
test `K4_1938WA...jpg` tidak membawa prefix di namanya, padahal
fotonya AB 1938 WA. Pemeriksaan konsistensi label terhadap nama
file untuk seluruh 614 gambar menghasilkan nol ketidakcocokan
selain file itu, tetapi pemeriksaan tersebut bersifat sirkular,
karena labelnya memang berasal dari nama file. Ia membuktikan
konsistensi, bukan kebenaran foto. Spot-check visual penuh sebesar
10 persen, yaitu 61 gambar, belum selesai; baru 7 foto yang dibaca
langsung dan semuanya cocok dengan labelnya. Jadi risiko label
salah terlihat rendah, tetapi belum tertutup. Angka-angka di
proyek ini juga hanya berlaku untuk dataset Roboflow ini. Apakah
modelnya bertahan pada foto HP di jalan, dengan jarak, sudut, dan
cahaya yang jauh lebih beragam, belum teruji.

## Menjalankan ulang

Kebutuhan dasarnya Python 3.10 ke atas, dan GPU sangat disarankan.
Training penuh tidak realistis di CPU. Instal dependensinya dengan:

```bash
pip install -r requirements.txt
```

Unduh dataset dari halaman Roboflow di atas, pilih format YOLOv11,
dan ekstrak ke `data/roboflow/` sehingga tersedia `data.yaml` serta
folder `train/`, `valid/`, dan `test/`. Dataset tidak di-commit ke
repo ini; folder `data/` memang di-ignore. Perlu diketahui juga
bahwa halaman sumber tidak menyediakan paper, metrik acuan, atau
bobot model. Baseline di proyek ini karena itu dibangun sendiri,
bukan dengan membandingkan angka ke paper sumber.

Setelah dataset tersedia, jalankan audit, relabel, dan baseline
secara berurutan:

```bash
python scripts/audit_dataset.py --root data/roboflow
python scripts/generate_audit_csv.py
python scripts/relabel_to_jogja.py
python scripts/baseline_pretrained.py \
    --test-dir data/roboflow/test/images \
    --out runs/baseline_pretrained.csv
```

Training dua kelas, di mesin bergpu atau lewat notebook Colab yang
ada di repo ini (`colab_yolo11_plat_jogja.ipynb`, runtime GPU T4):

```bash
python scripts/train.py --data configs/data_jogja_2cls.yaml \
    --model yolo11n.pt --epochs 100 --imgsz 960 --batch 16 --fliplr 0.0
```

Evaluasi standar setelah training:

```bash
yolo val model=runs/yolo11_plat/weights/best.pt \
    data=configs/data_jogja_2cls.yaml imgsz=960
```

Bobot hasil training Colab (`best.pt`, sekitar 5,6 MB) tidak ada di
repo ini, karena repo ini hanya berisi file teks. Untuk inference,
latih ulang dengan perintah di atas atau minta bobotnya dari
penulis. Angka test sebaiknya tidak dikutip tanpa konfigurasi dan
bobot yang sesuai.

## Struktur repo

```
README.md                      Dokumen ini
README_COLAB.md                Panduan menjalankan notebook di Colab
colab_yolo11_plat_jogja.ipynb  Notebook end-to-end: audit, relabel, train, evaluasi
configs/
  data_roboflow_1cls.yaml      Dataset asli satu kelas
  data_jogja_2cls.yaml         Dataset relabel dua kelas
docs/
  KEBUTUHAN_DATA.md            Spesifikasi data dunia nyata (Blok B)
  RELABEL.md                   Metode relabel dan batasannya
papers/
  DAFTAR_PUSTAKA.md            Daftar pustaka dan tabel perbandingan paper
scripts/
  audit_dataset.py             Audit dataset sumber
  baseline_pretrained.py       Baseline YOLOv11 pretrained COCO
  detect_classify.py           Kerangka pipeline deteksi + OCR (belum dievaluasi penuh)
  evaluate_detection.py        Confusion matrix Jogja vs non-Jogja
  generate_audit_csv.py        Audit label dari nama file
  relabel_to_jogja.py          Relabel satu kelas menjadi dua kelas
  train.py                     Training fine-tune YOLOv11
```

## Rencana lanjut

Pertama, selesaikan spot-check visual 10 persen. Daftarnya sudah
ada di `runs/spotcheck_sample.csv`. Bila ditemukan label yang salah,
perbaiki CSV auditnya dan jalankan ulang relabel, bukan menambal
angka di laporan.

Kedua, kumpulkan data dunia nyata, yang di dokumen proyek disebut
Blok B: 600 sampai 1.000 foto HP baru, 300 sampai 500 per kelas,
ditambah test khusus minimal 100 gambar dengan komposisi 50/50 dari
kendaraan, hari, dan lokasi yang berbeda dari train. Komposisinya
harus mencakup siang dan malam, plat depan dan belakang, dekat dan
jauh, motor dan mobil, serta plat hitam dan putih. Foto asli dari
HP disimpan apa adanya, tanpa flip, dan disertai CSV audit berisi
teks plat hasil baca manusia. Spesifikasi lengkapnya ada di
`docs/KEBUTUHAN_DATA.md`.

Ketiga, bangun pipeline OCR dua tahap dan bandingkan dengan model
dua kelas langsung pada test set yang sama, sebelum mengklaim
pembacaan huruf. Terakhir, training dari nol sengaja tidak
dilakukan. Bobot pretrained tersedia dan fine-tuning berhasil,
sehingga urutan yang dipakai adalah inference langsung, lalu
fine-tuning, lalu transfer learning, dan training dari nol hanya
bila tidak ada bobot yang bisa dipakai ulang.

## Referensi

Sejauh pencarian yang dilakukan, tidak ada paper yang secara khusus
mengklasifikasikan kode wilayah AB versus non-AB. Yang ada adalah
deteksi plat dan OCR plat Indonesia secara umum, serta klasifikasi
atribut plat lain seperti warna. Daftar berikut adalah pembanding
metode terdekat; tabel perbandingan lengkapnya ada di
`papers/DAFTAR_PUSTAKA.md`.

- Martadinata, Firdaus, dan Amal (2025). Detection and Identification
  of Vehicle License Plates in Indonesia Transportation System Based
  on Deep Learning Using YOLOv11 and EasyOCR. Jurnal Sisfokom, 14(4),
  573-578. DOI: 10.32736/sisfokom.v14i4.2524.
- Kurniawan dan Sari (2025). Automatic License Plate Detection System
  with YOLOv11 Algorithm. Journal of Applied Informatics and
  Computing, 9(6), 3097-3109.
- Wijaya dan Arymurthy (2025). Indonesian License Plate Detection and
  Recognition System using Gaussian YOLOv7. Jurnal Ilmu Komputer dan
  Informasi, 18(2), 141-153. DOI: 10.21609/jiki.v18i2.1320.
- Laroca dkk. (2018). A Robust Real-Time Automatic License Plate
  Recognition Based on the YOLO Detector. IJCNN 2018 (dataset
  UFPR-ALPR).

## Atribusi

Dataset sumber oleh Universitas Teknologi Yogyakarta, lisensi CC BY
4.0. Atribusi itu perlu dicantumkan bila datasetnya dipakai ulang.
Kode dan dokumen di repo ini dibuat untuk proyek ini; bobot model,
gambar dataset, dan laporan PDF final tidak termasuk di dalamnya.
