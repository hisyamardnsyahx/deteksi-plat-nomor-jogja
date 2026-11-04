"""Baseline A3: inference YOLOv11 pretrained COCO di data test.

PENTING: yolo11n.pt dilatih di COCO (80 kelas: mobil, motor, orang, ...).
COCO TIDAK punya kelas 'plat nomor'. Jadi ekspektasi yang benar:
model ini akan mendeteksi kendaraan, bukan plat — dan itu BUKAN
kegagalan eksperimen, itu bukti kenapa fine-tune wajib sebelum ada
klaim deteksi plat. Catat hasil mentahnya di CSV, jangan dibulatkan
jadi 'mirip paper' kalau tidak ada deteksi platnya.

Penggunaan:
    python scripts/baseline_pretrained.py --test-dir data/roboflow/test/images \
        --out runs/baseline_pretrained.csv
"""
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="runs/baseline_pretrained.csv")
    args = ap.parse_args()

    model = YOLO(args.model)
    print("Kelas COCO yang bisa diprediksi model pretrained:",
          list(model.names.values())[:10], "... total", len(model.names))
    rows = []
    for img in sorted(Path(args.test_dir).glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        res = model.predict(str(img), imgsz=args.imgsz, conf=args.conf,
                            verbose=False)[0]
        dets = [(model.names[int(c)], round(float(s), 3))
                for c, s in zip(res.boxes.cls, res.boxes.conf)] \
            if res.boxes is not None else []
        rows.append({"image": img.name, "n_det": len(dets),
                     "detections": ";".join(f"{n}:{s}" for n, s in dets)})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "n_det", "detections"])
        w.writeheader()
        w.writerows(rows)
    print(f"Tersimpan: {out} ({len(rows)} gambar)")
    print("Interpretasi: baris tanpa deteksi kelas plat = ekspektasi normal "
          "untuk model COCO. Langkah berikut: fine-tune (scripts/train.py).")


if __name__ == "__main__":
    main()
