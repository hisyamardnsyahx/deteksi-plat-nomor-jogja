# Daftar pustaka & tabel perbandingan — proyek plat Jogja vs non-Jogja

Disusun 4 Nov 2026 dari daftar paper/jurnal yang sudah diberikan ke Syam di side chat
"Classify Jogja license plates with YOLOv11" pada hari yang sama (16:10 WIB).
File ini adalah draf siap tempel ke laporan — bukan laporan final yang sudah direvisi.

## Catatan kejujuran yang harus ikut ke laporan

1. **Tidak ada satu pun paper di bawah ini yang mengklasifikasikan kode wilayah
   AB (Jogja) vs non-Jogja.** Yang ada: deteksi plat + OCR plat Indonesia secara
   umum, dan klasifikasi atribut plat lain (warna). Jadi bagian AB vs non-AB
   tetap celah kebaruan proyek ini.
2. **Perbandingan dengan paper Indonesia bersifat perbandingan metode, bukan
   reproduce persis.** Dari pengecekan langsung ke naskah pada 4 Nov 2026:
   SISFOKOM, JAIC, dan Gaussian YOLOv7 tidak memuat tautan dataset maupun code
   publik (pencarian "github" di naskah JAIC dan Gaussian YOLOv7 menghasilkan nol).
   Angka mereka tidak bisa direproduce persis — hanya metodenya yang bisa
   ditiru dengan data lain, persis seperti yang sudah dilakukan dengan dataset
   Roboflow UTY.
3. **Jalur OCR dua tahap belum dijalankan dan belum dibandingkan secara empiris
   di proyek ini.** Yang dievaluasi di laporan hanya YOLOv11 deteksi 2 kelas
   langsung (tanpa keluaran string teks plat).

## Tabel perbandingan (siap tempel)

| Paper | Metode | Dataset | Metrik headline | Relevansi ke proyek AB vs non-AB |
|---|---|---|---|---|
| SISFOKOM 2025 (Martadinata dkk.) | YOLOv11 deteksi + EasyOCR | 709 gambar plat motor/mobil Indonesia | mAP@50 94,2%; precision 97,7%; recall 86,7%; akurasi karakter EasyOCR 91,0% | Pembanding terdekat untuk jalur OCR dua tahap yang belum dijalankan |
| JAIC 2025 (Kurniawan & Sari) | YOLOv11 deteksi plat + karakter | 4.000 gambar plat + 3.000 gambar karakter | Precision 0,891; recall 0,911; mAP50 0,906; uji real-time 12/12 plat dikenali | Pembanding YOLOv11 Indonesia kedua; sumber datanya Kaggle + RoboFlow + 500 foto HP peneliti, nama dataset Kaggle persisnya tidak disebut |
| JIKI 2025 (Wijaya & Arymurthy) | Gaussian YOLOv7 + Spatial Transformer Network | Dataset ALPR Indonesia baru, koleksi sendiri (±1.000 gambar) | Tidak dikutip di sini — cek naskah sebelum menulis angka | Berguna bila plat miring di foto HP jadi masalah (STN meluruskan plat); dataset/code tidak publik |
| Pepadun 2026 | YOLOv5/v7/v8/v9 klasifikasi warna plat | 1.214 gambar | YOLOv9 terbaik: precision 97,20%; recall 96,50%; mAP50 98,10% | Analogi terkuat untuk ide klasifikasi atribut plat dengan YOLO — hanya atributnya warna (hitam, putih, kuning, merah, non-plat), bukan kode wilayah |
| RESTI 2025 | YOLOv8 deteksi + perbandingan OCR (EasyOCR vs TesseractOCR vs TensorFlow-CNN) | Gambar statis + video | EasyOCR 70%; TesseractOCR 68%; TensorFlow-CNN lebih unggul di video | Bukti bahwa pilihan OCR sangat memengaruhi hasil, bukan cuma detektornya |
| Laroca dkk. 2018 (IJCNN) — dasar UFPR-ALPR | YOLO detector untuk ALPR | UFPR-ALPR: 4.500 gambar, 150 kendaraan, 3 kamera | Lihat naskah — jangan kutip angka dari abstrak saja | Satu-satunya target reproduce yang sah dengan dataset + code: dataset lewat formulir permintaan (akademik, non-komersial), code di repo penulis; kelemahannya plat Brasil, jadi hanya metodenya yang bisa direproduce, bukan klasifikasi AB |
| **Proyek ini (Colab, Tesla T4)** | **YOLOv11n deteksi 2 kelas langsung (plat-jogja / plat-non-jogja), imgsz 960, 100 epoch** | **Roboflow UTY: 614 gambar (train 491 / valid 62 / test 61), label dari nama file + verifikasi visual terbatas** | **Test: mAP50 0,9950; mAP50-95 0,8002; precision 0,9968; recall 1,0000; 33 Jogja + 28 non-Jogja benar, 0 salah kelas, 5 false positive** | **Satu-satunya baris yang mengklasifikasikan AB vs non-AB; caveat: label diturunkan dari nama file, spot-check visual 10% masih outstanding sebelum klaim final** |

## Daftar pustaka

1. Martadinata, Firdaus, Amal (2025). *Detection and Identification of Vehicle
   License Plates in Indonesia Transportation System Based on Deep Learning
   Using YOLOv11 and EasyOCR*. Jurnal Sisfokom, Vol. 14 No. 4, hlm. 573–578.
   DOI: 10.32736/sisfokom.v14i4.2524.
   PDF: https://jurnal.atmaluhur.ac.id/index.php/sisfokom/article/download/2524/1376/10230
2. Kurniawan & Sari (2025). *Automatic License Plate Detection System with
   YOLOv11 Algorithm*. Journal of Applied Informatics and Computing,
   Vol. 9 No. 6, hlm. 3097–3109.
   PDF: https://jurnal.polibatam.ac.id/index.php/JAIC/article/download/11484/3171
3. Wijaya & Arymurthy (2025). *Indonesian License Plate Detection and
   Recognition System using Gaussian YOLOv7*. Jurnal Ilmu Komputer dan
   Informasi, Vol. 18 No. 2, hlm. 141–153. DOI: 10.21609/jiki.v18i2.1320.
   PDF: https://repository.horizon.ac.id/files/original/f92de31e491b6421ee88d39a25695503.pdf
4. *Assessing Detection and Classification Performance for Vehicle License
   Plate Colors Using YOLOv5, YOLOv7, YOLOv8, and YOLOv9*. Jurnal Pepadun,
   Vol. 7 No. 1, 2026, hlm. 67–79.
   PDF: https://pepadun.fmipa.unila.ac.id/index.php/jurnal/article/download/349/210/2004
5. *Automated Indonesian Plate Recognition: YOLOv8 Detection and
   TensorFlow-CNN Character Classification*. Jurnal RESTI, 2025.
   DOI: 10.29207/resti.v9i3.6310.
   Halaman artikel (PDF lewat tombol unduh di halaman itu):
   https://jurnal.iaii.or.id/index.php/RESTI/article/view/6310
6. Laroca dkk. (2018). *A Robust Real-Time Automatic License Plate Recognition
   Based on the YOLO Detector*. IJCNN 2018.
   PDF: https://raysonlaroca.github.io/presentations/laroca2018robust.pdf

## Tidak open access penuh — jangan diklaim sudah dibaca penuh

- Paper YOLOv11 + MobileViTv2 (Springer, ICDAM 2025): halaman publik hanya
  menampilkan abstrak; naskah lengkap butuh akses institusi/UGM.
  https://doi.org/10.1007/978-3-032-03751-0_32

## Yang masih perlu dilakukan Syam (bukan dikerjakan diam-diam)

- [ ] Unduh ke-6 PDF di atas ke folder `papers/` di proyek ini (tautan sudah
      terverifikasi sebagai tautan yang diberikan 4 Nov 2026; unduhan ulang
      belum dilakukan di run ini).
- [ ] Tempel tabel perbandingan + daftar pustaka ini ke laporan, lalu sesuaikan
      format sitasi dengan gaya yang dipakai laporan (penulis Pepadun dan RESTI
      belum dilengkapi di atas karena tidak tercantum di daftar yang diberikan —
      lengkapi dari naskah, jangan ditebak).
- [ ] Putuskan: cukup tabel ini, atau jalankan juga jalur OCR dua tahap
      (YOLO 1 kelas + EasyOCR) sebagai pembanding empiris — jalur itu yang
      membuat perbandingan dengan SISFOKOM benar-benar sepadan.
