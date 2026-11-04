"""Audit dataset YOLO: hitung gambar, label, kelas, ukuran kotak.

Penggunaan:
    python scripts/audit_dataset.py --data data/roboflow/data.yaml
    python scripts/audit_dataset.py --root data/roboflow
Tanpa dataset (folder data/roboflow kosong) skrip berhenti dengan pesan
yang jelas, bukan angka palsu.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path


def audit_split(images_dir: Path):
    labels_dir = images_dir.parent / "labels"
    counts = Counter()
    box_w, box_h, missing, empty = [], [], 0, 0
    for img in sorted(images_dir.glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label = labels_dir / (img.stem + ".txt")
        if not label.exists():
            missing += 1
            continue
        lines = [ln.strip() for ln in label.read_text().splitlines() if ln.strip()]
        if not lines:
            empty += 1
        for ln in lines:
            parts = ln.split()
            counts[int(parts[0])] += 1
            box_w.append(float(parts[3]))
            box_h.append(float(parts[4]))
    n_img = len([p for p in images_dir.glob("*")
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    return {
        "images": n_img, "boxes": sum(counts.values()),
        "per_class": dict(counts), "missing_labels": missing,
        "empty_labels": empty,
        "mean_box_w": sum(box_w) / len(box_w) if box_w else 0,
        "mean_box_h": sum(box_h) / len(box_h) if box_h else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/roboflow")
    ap.add_argument("--data", default=None, help="opsional: path data.yaml")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.exists() or not any(root.rglob("*.jpg")):
        print(f"Dataset belum ada di {root}. Unduh ZIP YOLOv11 dari Roboflow "
              "dan ekstrak ke folder itu dulu (lihat README bagian 5).")
        sys.exit(2)
    for split in ("train", "valid", "test"):
        d = root / split / "images"
        if d.exists():
            print(split, audit_split(d))
        else:
            print(split, "tidak ditemukan (cek nama folder: valid vs val)")


if __name__ == "__main__":
    main()
