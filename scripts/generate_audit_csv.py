"""Buat CSV audit Jogja/non-Jogja dari nama file + verifikasi manual.

Temuan audit 4 Nov 2026: nama file Roboflow mengkodekan teks plat
(mis. K1_AB2430VL = plat AB 2430 VL, terverifikasi visual pada 3 sampel).
TAPI nama file bukan ground truth sempurna: 1 file test
(K4_1938WA) tidak membawa prefix huruf di namanya, padahal plat di
fotonya AB 1938 WA (terverifikasi visual). Karena itu skrip ini:
  - mem-parse prefix huruf di awal token plat,
  - menandai token yang TIDAK diawali huruf sebagai manual_review,
  - TIDAK pernah menebak label untuk manual_review.

Keluaran: CSV image,prefix,label,source,notes.
Jalankan lalu periksa baris manual_review satu per satu sebelum relabel.
"""
import csv
import re
import sys
from pathlib import Path

# Verifikasi visual manual (4 Nov 2026, dibaca langsung dari fotonya):
MANUAL = {
    "K4_1938WA_jpg.rf.e572eadcb5e7b50790ef081d9f4370d1.jpg": ("AB", "plat-jogja"),
}


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("data/roboflow")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        Path("runs/audit_jogja.csv")
    rows = []
    for split in ("train", "valid", "test"):
        for img in sorted((root / split / "images").glob("*")):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if img.name in MANUAL:
                prefix, label = MANUAL[img.name]
                rows.append({"image": img.name, "split": split,
                             "prefix": prefix, "label": label,
                             "source": "manual_visual",
                             "notes": "nama file tanpa prefix huruf; "
                                      "prefix dibaca dari foto"})
                continue
            m = re.match(r"^[A-Z0-9]+_([A-Z]+)\d", img.name)
            if not m:
                rows.append({"image": img.name, "split": split,
                             "prefix": "", "label": "manual_review",
                             "source": "filename",
                             "notes": "token plat tidak diawali huruf; "
                                      "periksa foto sebelum memberi label"})
                continue
            prefix = m.group(1)
            rows.append({"image": img.name, "split": split,
                         "prefix": prefix,
                         "label": "plat-jogja" if prefix == "AB"
                         else "plat-non-jogja",
                         "source": "filename",
                         "notes": ""})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "split", "prefix",
                                          "label", "source", "notes"])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"Ditulis {out}: {len(rows)} baris")
    print("Per split x label:", Counter((r["split"], r["label"])
                                        for r in rows))
    print("Manual review tersisa:",
          [r["image"] for r in rows if r["label"] == "manual_review"])


if __name__ == "__main__":
    main()
