# Deteksi Plat Nomor: Jogja vs Non-Jogja (YOLOv11)

Tujuan: dari foto kendaraan, jawab **Jogja (kode wilayah AB) atau non-Jogja**.
Kode AB = Daerah Istimewa Yogyakarta (Kota Yogyakarta, Sleman, Bantul,
Kulon Progo, Gunungkidul). Semua kode lain = non-Jogja.

## 1. Temuan baseline (langkah A) — dibaca dari sumber, 4 Nov 2026

Sumber: Roboflow Universe, Universitas Teknologi Yogyakarta,
“Deteksi Plat Nomor Kendaraan 2”, dataset v1, 16 Des 2023.
Lisensi: CC BY 4.0. File yang dipakai: ZIP YOLOv11 dari Google Drive
Syam, `Deteksi Plat Nomor Kendaraan 2.v1i.yolov11.zip` (53 MB).

| Item | Nilai terverifikasi dari file yang diunduh |
|---|---|
| Total gambar | 614 |
| Split bawaan | train 491 (80%), valid 62 (10%), test 61 (10%) |
| Kelas asli (`data.yaml`) | 1 kelas: `Plat-Nomor` |
| Kotak anotasi | train 492, valid 62, test 61 — train berisi 1 kotak degeneratif 0,002×0,002 (noise, dibuang saat relabel) sehingga train bersih = 491 kotak |
| Ukuran gambar | 720×720 (cek 5 gambar test, semua RGB 720×720) |
| Ukuran kotak rata-rata | lebar ~0,33, tinggi ~0,11 (fraksi gambar) — plat relatif besar di frame, bagus untuk deteksi dan OCR |
| Preprocessing (halaman sumber) | Auto-Orient, Resize: Stretch to 720×720, tanpa augmentasi |
| Paper / metrik acuan / bobot model | **Tidak ada di halaman sumber** |
| Format unduhan yang tersedia | YOLOv11, YOLOv12, YOLOv8, COCO, Pascal VOC, dll. |

### Kesimpulan A3 yang jujur

**Tidak bisa “reproduce results” dalam arti membandingkan mAP dengan
paper/model sumber, karena tidak ada angka acuan yang dipublikasikan di
halaman tersebut.** Ini dataset saja, bukan paper + bobot model.

Konsekuensi logisnya, mengikuti alur tugas (A3a):

1. Cek code → tidak ada repo/code training di halaman sumber.
2. Cek data → satu kelas deteksi plat, bukan dua kelas Jogja/non-Jogja.
3. Jadi baseline yang bisa direproduksi di sini adalah:
   - **Baseline detektor:** YOLOv11 pretrained COCO → inference di data test
     → dicatat hasilnya (lihat `scripts/baseline_pretrained.py`).
   - **Baseline terlatih:** fine-tune YOLOv11 dari `yolo11n.pt` pada
     614 gambar itu → ukur mAP50 / mAP50-95 di valid/test bawaan.
   - Keduanya **belum menjawab Jogja vs non-Jogja**. Itu baru langkah A4/B.

### Temuan audit yang mengubah langkah relabel (penting!)

Nama file Roboflow ternyata mengkodekan teks plat, mis.
`K1_AB2430VL_...jpg` = plat AB 2430 VL. Tiga sampel dicek visual dan
cocok persis dengan plat di foto (AB 2430 VL, AA 4072 LB, AB 1938 WA).
Satu pengecualian membuktikan nama file bukan ground truth sempurna:
`K4_1938WA_...jpg` tidak membawa prefix di namanya, padahal fotonya
AB 1938 WA — file ini ditangani manual, bukan ditebak.

Hasil audit dari nama file (CSV lengkap: `runs/audit_jogja.csv`,
dibuat `scripts/generate_audit_csv.py`):

| Split | Jogja (AB) | Non-Jogja | Total |
|---|---|---|---|
| train | 266 (54,2%) | 225 | 491 |
| valid | 30 (48,4%) | 32 | 62 |
| test | 33 (54,1%) | 28 | 61 |
| **total** | **329** | **285** | **614** |

Dua cek kebersihan split juga lolos: **nol token plat yang sama
muncul di dua split berbeda, dan nol duplikat token dalam satu split**.
Jadi split asli dipertahankan untuk dataset 2 kelas
(`data/jogja_2cls/`, dibuat `scripts/relabel_to_jogja.py`) — tidak
perlu diacak ulang. Catatan kejujuran: label 2 kelas ini diturunkan
dari nama file + 1 verifikasi visual, bukan dari pembacaan manual
614 foto. Sebelum dipakai sebagai klaim final di laporan, lakukan
spot-check visual acak minimal 10% (62 foto) dan catat hasilnya;
kalau ada yang salah baca, perbaiki CSV-nya dan jalankan ulang relabel.

### Kenapa detektor 1 kelas tidak cukup untuk Jogja vs non-Jogja

Plat Jogja dan non-Jogja bentuknya sama. Bedanya hanya huruf kode
wilayah di awal plat (AB vs yang lain), plus kadang warna plat
(hitam lama vs putih baru — itu bukan penanda wilayah).
Jadi klasifikasi Jogja harus membaca huruf, bukan sekadar bentuk kotak plat.

### Arsitektur yang direkomendasikan (A4)

**Pipeline 2 tahap (utama, paling bisa diaudit):**

```
foto -> YOLOv11 deteksi 1 kelas "plat" -> crop plat (+padding 5-10%)
      -> OCR crop -> baca prefix -> AB = jogja, selain AB = non-jogja
      -> kalau OCR confidence rendah: label "tidak yakin", jangan dipaksa
```

**Eksperimen pembanding (kalau dosen minta murni YOLO):**
YOLOv11 deteksi 2 kelas langsung: `plat-jogja`, `plat-non-jogja`.
Ini satu model, tapi butuh data jauh lebih banyak dan `imgsz` besar,
karena model harus belajar bentuk huruf AB vs huruf lain dari kotak
kecil. Laporkan keduanya, jangan hanya yang menang.

## 2. Model & hyperparameter (A1, A4)

- Framework: Ultralytics YOLOv11 (`ultralytics`).
- Arsitektur: YOLO11n backbone C3k2 + SPPF + C2PSA, neck PAN, head Detect
  P3/8–P5/32, anchor-free, loss IoU+DFL untuk box dan BCE untuk klas.
- Bobot awal: `yolo11n.pt` (pretrained COCO). Naik ke `yolo11s.pt` hanya
  kalau mAP val mentok dan masih ada headroom GPU/waktu.
  Perkiraan ukuran (Ultralytics): n ≈ 2,6 jt parameter / 6,5 GFLOPs,
  s ≈ 9,4 jt / 21,5 GFLOPs.
- `imgsz`: 640 untuk baseline detektor; **960 untuk tugas Jogja**
  (plat kecil, huruf harus terbaca). 1280 hanya kalau GPU kuat.
- `epochs`: 100 dulu, `patience=50` (early stopping Ultralytics).
- `batch`: 16 di GPU T4 pada imgsz 640; turunkan ke 8 di imgsz 960/1280.
- `lr0=0.01`, optimizer SGD default Ultralytics, `momentum=0.937`,
  `weight_decay=0.0005`, `warmup_epochs=3`, cosine LR.
- Augmentasi default Ultralytics dipakai, **dengan dua koreksi penting
  untuk tugas ini**:
  - `fliplr=0.0` dan `flipud=0.0` — flip horizontal membalik huruf
    (AB jadi cermin), merusak pembelajaran huruf dan OCR.
  - `mosaic=1.0`, `close_mosaic=10` (mosaic dimatikan 10 epoch terakhir).
- `seed=42`, split bawaan Roboflow **jangan diacak ulang** untuk baseline.
  Untuk data Jogja baru, pakai stratified split + group split
  (foto kendaraan yang sama tidak boleh bocor ke train dan test).
- Evaluasi: `model.val()` standar untuk mAP50 dan mAP50-95
  (conf validasi default 0.001, jangan pakai threshold deployment untuk
  headline mAP). Precision/Recall/F1 dilaporkan terpisah pada
  operating point yang dipilih (mis. conf=0.25–0.35), plus confusion
  matrix Jogja vs non-Jogja vs tidak-yakin untuk pipeline OCR.

Perintah baseline (setelah dataset diunduh ke `data/roboflow/`):

```bash
# venv proyek (sudah disiapkan): .venv/bin/python
.venv/bin/python scripts/audit_dataset.py --root data/roboflow
.venv/bin/python scripts/baseline_pretrained.py \
    --test-dir data/roboflow/test/images \
    --out runs/baseline_pretrained.csv
.venv/bin/python scripts/train.py --data data/roboflow/data.yaml \
    --model yolo11n.pt --epochs 100 --imgsz 640 --batch 16
```

Perintah 2 kelas (dataset relabel sudah jadi di `data/jogja_2cls/`):

```bash
.venv/bin/python scripts/train.py --data data/jogja_2cls/data.yaml \
    --model yolo11n.pt --epochs 100 --imgsz 960 --batch 8 --fliplr 0.0
```

## 3. Data yang dibutuhkan (langkah B)

Detail lengkap: `docs/KEBUTUHAN_DATA.md`. Ringkasnya:

| Kebutuhan | Target |
|---|---|
| Data baru foto HP | 600–1.000 gambar (300–500 Jogja, 300–500 non-Jogja), seimbang |
| Test set khusus Jogja | min. 100 gambar (50/50), dari kendaraan/hari/lokasi berbeda dari train |
| Dataset Roboflow | dipakai sebagai pretrain detektor, tapi tiap gambarnya harus diaudit dulu: apakah platnya AB atau bukan — label aslinya tidak menyimpan info itu |
| Komposisi | siang/malam, depan/belakang, dekat/jauh, miring, motor/mobil, plat hitam & putih, plat kotor/blur ringan |
| Anotasi | kotak ketat mengelilingi plat saja (bukan seluruh kendaraan), format YOLO txt + CSV teks plat hasil baca manusia untuk audit OCR |

## 4. Alur B5 — keputusan transfer learning

1. Inference pretrained / detektor Roboflow di data baru → catat.
2. Kalau deteksi plat sudah bagus tapi klasifikasi Jogja salah →
   masalahnya di tahap baca huruf, bukan detektor: perbaiki OCR/crop,
   bukan langsung training ulang detektor.
3. Kalau deteksi plat jelek di foto HP (jarak/sudut/cahaya berbeda) →
   fine-tune detektor dari bobot Roboflow dengan data baru
   (transfer learning), bukan training dari nol.
4. Training dari nol hanya kalau tidak ada bobot yang bisa dipakai ulang
   — di sini tidak berlaku, karena `yolo11n.pt` dan bobot hasil
   fine-tune Roboflow tersedia.

## 5. Cara mengunduh dataset sumber

Di halaman Roboflow: Download → pilih format YOLOv11 → Download ZIP.
Butuh akun Roboflow gratis untuk ekspor. Ekstrak ke
`data/roboflow/` sehingga ada `data.yaml`, `train/`, `valid/`, `test/`.
Jangan commit dataset ke repo — folder `data/` di-ignore.

## 6. Batasan lingkungan di VM ini

VM ini (cek 4 Nov 2026): 4 CPU, RAM 7,7 GB, **tanpa GPU, tanpa torch
terinstal**. Training penuh di sini tidak realistis untuk 100 epoch.
Jalankan training di Colab (GPU T4 gratis) atau laptop/PC bergpu,
dengan perintah yang sama di atas. Di VM ini yang realistis: audit
dataset, skrip, dan inference ringan setelah bobot jadi.
