# Deteksi Plat Nomor: Jogja vs Non-Jogja dengan YOLOv11

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-green)
![Dataset](https://img.shields.io/badge/Dataset-Roboflow%20UTY%20v1-orange)

Mengklasifikasikan plat nomor kendaraan Indonesia menjadi **Jogja** atau
**non-Jogja** dari foto kendaraan, menggunakan YOLOv11. Plat Jogja
didefinisikan oleh kode wilayah **AB** (Daerah Istimewa Yogyakarta: Kota
Yogyakarta, Sleman, Bantul, Kulon Progo, Gunungkidul). Semua kode wilayah
lain diklasifikasikan sebagai non-Jogja.

> **Ide kuncinya:** plat AB dan non-AB bentuknya sama, jadi klasifikasi
> tidak bisa mengandalkan bentuk kotak plat saja. Perbedaannya ada pada
> huruf prefix di awal plat — model harus belajar pola visual huruf
> tersebut, atau membacanya lewat OCR.

## Hasil Utama

Model yang dilatih dan dievaluasi di sini adalah **YOLOv11n deteksi
2 kelas langsung** (`plat-jogja` / `plat-non-jogja`), di-fine-tune dari
`yolo11n.pt` (pretrained COCO) di Google Colab (GPU Tesla T4, 100 epoch,
~24 menit, tanpa early stopping).

| Split | Gambar | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Validasi | 62 | 0,985 | 0,978 | 0,995 | 0,787 |
| Test | 61 | 0,997 | 1,000 | 0,995 | 0,800 |

Per kelas pada validasi: `plat-jogja` (30 instance) precision 0,97,
recall 1,000, mAP50 0,995; `plat-non-jogja` (32 instance) precision 1,000,
recall 0,956, mAP50 0,995.

Confusion matrix pada test (IoU ≥ 0,5, confidence 0,25): **33 Jogja
benar, 28 non-Jogja benar, 0 salah kelas, 0 terlewat, 5 false positive**
(kotak tambahan, bukan kesalahan AB vs non-AB). Seluruh 61 plat test
ditemukan dan diklasifikasikan dengan benar.

Sebagai pembanding, model pretrained COCO tanpa fine-tuning menghasilkan
**0 deteksi plat** pada 61 gambar test (284 deteksi kendaraan/orang:
motor 175, mobil 90, orang 16, truk 1, pot tanaman 1, ransel 1) — wajar,
karena COCO tidak memiliki kelas plat nomor. Fine-tuning wajib sebelum
ada klaim deteksi plat.

## Metode

Dua jalur dibedakan secara tegas di proyek ini:

1. **Jalur yang dilatih dan dievaluasi (hasil di atas).** YOLOv11
   deteksi 2 kelas langsung. Model mengeluarkan kotak, label, dan
   confidence — bukan string teks plat. `imgsz=960` dipilih agar huruf
   prefix cukup terbaca secara visual oleh model.
2. **Jalur rekomendasi (belum dievaluasi empiris).** Deteksi plat
   1 kelas → crop dengan padding 5–10% → OCR membaca prefix → AB =
   Jogja, selain AB = non-Jogja, OCR tidak yakin = `tidak yakin`.
   Jalur ini lebih mudah diaudit per huruf dan merupakan langkah lanjut
   yang disarankan, tetapi angkanya belum boleh diklaim dari proyek ini.
   Skrip kerangkanya ada di `scripts/detect_classify.py`.

### Konfigurasi training (Colab, Tesla T4)

| Parameter | Nilai |
|---|---|
| Model awal | `yolo11n.pt` (Ultralytics YOLOv11n, backbone C3k2 + SPPF + C2PSA, neck PAN, head Detect anchor-free) |
| Data | `configs/data_jogja_2cls.yaml` (2 kelas) |
| Epoch / patience / seed | 100 / 50 / 42 |
| Image size / batch / workers | 960 / 16 / 2 |
| Optimizer / LR awal | SGD default Ultralytics / 0,01 (momentum 0,937, weight decay 0,0005, warmup 3 epoch) |
| Augmentasi kunci | `mosaic=1.0`, `close_mosaic=10`, **`fliplr=0.0`, `flipud=0.0`** |

Flip horizontal/vertikal **wajib dimatikan**: flip mencerminkan huruf
plat dan merusak pembelajaran prefix maupun OCR. Evaluasi headline
menggunakan `model.val()` standar; confusion dihitung pada IoU ≥ 0,5
dan confidence 0,25.

## Dataset

Sumber: [Deteksi Plat Nomor Kendaraan 2, Universitas Teknologi
Yogyakarta, Roboflow Universe v1 (16 Des 2023)](https://universe.roboflow.com/universitas-teknologi-yogyakarta-8owbv/deteksi-plat-nomor-kendaraan-2/dataset/1),
lisensi **CC BY 4.0**. Preprocessing sumber: Auto-Orient dan Resize
Stretch ke 720×720, tanpa augmentasi.

| Item | Nilai |
|---|---|
| Total gambar | 614 (train 491 / valid 62 / test 61) |
| Kelas asli | 1 kelas: `Plat-Nomor` |
| Kotak anotasi | train 492, valid 62, test 61 |
| Ukuran kotak rata-rata | lebar ~0,33, tinggi ~0,11 (fraksi gambar) |

### Relabel menjadi 2 kelas

Dataset asli hanya memiliki 1 kelas, sehingga label Jogja/non-Jogja
dibuat dengan relabel (`scripts/relabel_to_jogja.py`, detail di
`docs/RELABEL.md`). Nama file Roboflow mengkodekan teks plat
(mis. `K1_AB2430VL_...jpg` = AB 2430 VL), dan relabel memanfaatkan itu.
Split asli dipertahankan agar tidak terjadi kebocoran token plat antar
split (cek: nol token yang sama muncul di dua split berbeda, nol
duplikat dalam satu split). Satu kotak train degeneratif (~0,002×0,002)
dibuang sebagai noise anotasi, sehingga train bersih = 491 kotak.

| Split | Jogja (AB) | Non-Jogja | Total |
|---|---:|---:|---:|
| Train | 266 | 225 | 491 |
| Valid | 30 | 32 | 62 |
| Test | 33 | 28 | 61 |
| **Total** | **329** | **285** | **614** |

### Keterbatasan label — dibaca sebelum mengutip angka

- Label 2 kelas **diturunkan dari nama file, bukan dari pembacaan
  manual 614 foto**. Satu pengecualian sudah dikoreksi manual:
  file test `K4_1938WA...jpg` tidak membawa prefix di namanya, padahal
  fotonya AB 1938 WA.
- Cek konsistensi label-vs-nama-file untuk 614 gambar: 0 ketidakcocokan
  selain K4 tersebut. Cek ini bersifat sirkular (label memang berasal
  dari nama file), jadi ia membuktikan konsistensi, bukan kebenaran foto.
- Spot-check visual penuh 10% (61 gambar) **belum selesai**: 7 foto
  sudah dibaca visual dan semuanya cocok dengan labelnya. Karena itu
  risiko label salah tampak rendah, tetapi caveat ini belum tertutup.
- Angka di atas berlaku untuk dataset Roboflow ini, yang platnya
  relatif besar di frame. Generalisasi ke foto HP dunia nyata belum
  teruji — lihat [Langkah Lanjut](#langkah-lanjut).

## Reproduksi

### Prasyarat

- Python 3.10+, GPU disarankan (training penuh tidak realistis di CPU).
- Instal dependensi:

```bash
pip install -r requirements.txt
```

### 1. Unduh dataset

Dari halaman Roboflow di atas: Download → format YOLOv11 → ekstrak ke
`data/roboflow/` sehingga tersedia `data.yaml`, `train/`, `valid/`,
`test/`. Dataset **tidak** di-commit ke repo ini (folder `data/`
di-ignore); halaman sumber tidak menyediakan paper, metrik acuan,
maupun bobot model, sehingga baseline di sini dibangun sendiri, bukan
dengan membandingkan angka ke paper sumber.

### 2. Audit, relabel, dan baseline

```bash
python scripts/audit_dataset.py --root data/roboflow
python scripts/generate_audit_csv.py   # -> runs/audit_jogja.csv
python scripts/relabel_to_jogja.py     # -> data/jogja_2cls/
python scripts/baseline_pretrained.py \
    --test-dir data/roboflow/test/images \
    --out runs/baseline_pretrained.csv
```

### 3. Training

Di lokal (GPU) atau lewat notebook Colab yang tersedia di repo ini
(`colab_yolo11_plat_jogja.ipynb`, runtime GPU T4):

```bash
python scripts/train.py --data configs/data_jogja_2cls.yaml \
    --model yolo11n.pt --epochs 100 --imgsz 960 --batch 16 --fliplr 0.0
```

Evaluasi standar setelah training:

```bash
yolo val model=runs/yolo11_plat/weights/best.pt \
    data=configs/data_jogja_2cls.yaml imgsz=960
```

### 4. Bobot model

Bobot hasil training Colab (`best.pt`, ~5,6 MB) **tidak** ada di repo
ini, karena snapshot GitHub pada proyek ini hanya mencakup file teks.
Untuk inference, latih ulang dengan perintah di atas atau minta bobot
dari penulis. Jangan mengutip angka test tanpa bobot/konfigurasi yang
sesuai.

## Struktur Repo

```
├── README.md                      # Dokumen ini
├── README_COLAB.md                # Panduan menjalankan notebook di Colab
├── colab_yolo11_plat_jogja.ipynb  # Notebook end-to-end (GPU, audit, relabel, train, eval)
├── configs/
│   ├── data_roboflow_1cls.yaml    # Dataset asli 1 kelas
│   └── data_jogja_2cls.yaml       # Dataset relabel 2 kelas
├── docs/
│   ├── KEBUTUHAN_DATA.md          # Spesifikasi data dunia nyata (Blok B)
│   └── RELABEL.md                 # Metode dan caveat relabel
├── papers/
│   └── DAFTAR_PUSTAKA.md          # Daftar pustaka & tabel perbandingan paper
└── scripts/
    ├── audit_dataset.py           # Audit dataset sumber
    ├── baseline_pretrained.py     # Baseline YOLOv11 pretrained COCO
    ├── detect_classify.py         # Kerangka pipeline deteksi + OCR (belum dievaluasi penuh)
    ├── evaluate_detection.py      # Confusion matrix Jogja vs non-Jogja
    ├── generate_audit_csv.py      # Audit label dari nama file
    ├── relabel_to_jogja.py        # Relabel 1 kelas -> 2 kelas
    └── train.py                   # Training fine-tune YOLOv11
```

## Langkah Lanjut

1. **Selesaikan spot-check visual 10%** (61 gambar, daftar di
   `runs/spotcheck_sample.csv`); koreksi `runs/audit_jogja.csv` dan
   jalankan ulang relabel bila ditemukan label salah.
2. **Blok B — data dunia nyata:** kumpulkan 600–1.000 foto HP (300–500
   per kelas) plus test khusus min. 100 gambar (50/50) dari
   kendaraan/hari/lokasi berbeda; komposisi siang/malam, depan/belakang,
   dekat/jauh, motor/mobil, plat hitam/putih. Simpan foto asli (jangan
   diperkecil dulu), tanpa flip, disertai CSV audit teks plat hasil
   baca manusia. Detail lengkap di `docs/KEBUTUHAN_DATA.md`.
3. **Bangun dan bandingkan pipeline OCR dua tahap** pada test set yang
   sama, sebelum mengklaim pembacaan huruf.
4. Training dari nol sengaja tidak dilakukan: bobot pretrained tersedia
   dan fine-tuning/transfer learning berhasil, sesuai prioritas
   direct inference → fine-tuning → transfer learning → training dari
   nol.

## Referensi

Tidak ada paper yang ditemukan secara spesifik mengklasifikasikan kode
wilayah AB vs non-AB; daftar berikut adalah pembanding metode terdekat
(detail dan tabel perbandingan lengkap di `papers/DAFTAR_PUSTAKA.md`):

- Martadinata, Firdaus, & Amal (2025). *Detection and Identification of
  Vehicle License Plates in Indonesia Transportation System Based on
  Deep Learning Using YOLOv11 and EasyOCR*. Jurnal Sisfokom, 14(4),
  573–578. DOI: 10.32736/sisfokom.v14i4.2524.
- Kurniawan & Sari (2025). *Automatic License Plate Detection System
  with YOLOv11 Algorithm*. Journal of Applied Informatics and Computing,
  9(6), 3097–3109.
- Wijaya & Arymurthy (2025). *Indonesian License Plate Detection and
  Recognition System using Gaussian YOLOv7*. Jurnal Ilmu Komputer dan
  Informasi, 18(2), 141–153. DOI: 10.21609/jiki.v18i2.1320.
- Laroca et al. (2018). *A Robust Real-Time Automatic License Plate
  Recognition Based on the YOLO Detector*. IJCNN 2018 (dataset
  UFPR-ALPR).

## Atribusi

Dataset sumber oleh Universitas Teknologi Yogyakarta, lisensi CC BY 4.0
— cantumkan atribusi tersebut bila memakai ulang datasetnya. Kode dan
dokumen di repo ini dibuat untuk proyek ini; bobot model, dataset
gambar, dan laporan PDF final tidak termasuk dalam repo ini.
