"""Pipeline 2 tahap: deteksi plat -> crop -> OCR -> Jogja/non-Jogja.

Ini kerangka yang siap diisi OCR-nya (EasyOCR/PaddleOCR) setelah bobot
detektor jadi. Aturan klasifikasi sengaja eksplisit dan bisa diaudit:
  prefix == 'AB'            -> jogja
  prefix lain, OCR yakin    -> non-jogja
  OCR tidak yakin / kosong  -> tidak-yakin (bukan dipaksa salah satu)
Jangan menghapus kategori tidak-yakin dari laporan — itu metrik
kejujuran pipeline, bukan kegagalan yang disembunyikan.

Penggunaan (setelah instal easyocr dan bobot detektor jadi):
    python scripts/detect_classify.py --weights runs/yolo11_plat/weights/best.pt \
        --input data/jogja_2cls/test/images --out runs/pipeline_test.csv
"""
import argparse
import csv
import re
from pathlib import Path

# import cv2  # aktifkan saat OCR dipasang
# from ultralytics import YOLO
# import easyocr


def classify_prefix(text: str):
    """Kembalikan (label, prefix) dari teks OCR mentah."""
    clean = re.sub(r"[^A-Z0-9]", "", text.upper())
    m = re.match(r"^([A-Z]{1,2})", clean)
    if not m:
        return "tidak-yakin", ""
    prefix = m.group(1)
    return ("jogja" if prefix == "AB" else "non-jogja"), prefix


if __name__ == "__main__":
    # Self-test aturan klasifikasi (tanpa model/OCR):
    for txt, want in [("AB 1234 XY", "jogja"), ("B 5678 CD", "non-jogja"),
                      ("AA 1111 AA", "non-jogja"), ("", "tidak-yakin")]:
        got, _ = classify_prefix(txt)
        assert got == want, (txt, got, want)
    print("Aturan prefix lolos self-test. "
          "Pasang YOLO + OCR di skrip ini sebelum dipakai di data nyata.")
