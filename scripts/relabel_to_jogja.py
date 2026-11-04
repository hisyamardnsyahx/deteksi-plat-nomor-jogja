"""Relabel 1 kelas -> 2 kelas Jogja/non-Jogja dari CSV audit manual.

CSV wajib: image,split,prefix,label
  label: plat-jogja | plat-non-jogja | unreadable | manual_review
  prefix: AB untuk Jogja, kode lain untuk non-Jogja.
Baris unreadable/manual_review dibuang (bukan ditebak).
Split ASLI dipertahankan (train/valid/test sama seperti sumber),
karena audit menunjukkan split asli sudah seimbang Jogja/non-Jogja
dan tidak ada token plat yang bocor antar split. Opsi --reshuffle
hanya dipakai kalau split asli terbukti tidak seimbang.

Penggunaan:
    python scripts/relabel_to_jogja.py --csv runs/audit_jogja.csv \
        --src data/roboflow --dst data/jogja_2cls
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

MAP = {"plat-jogja": 0, "plat-non-jogja": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--src", default="data/roboflow")
    ap.add_argument("--dst", default="data/jogja_2cls")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reshuffle", action="store_true",
                    help="hanya kalau split asli tidak seimbang/bocor")
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.csv).open()))
    kept = [r for r in rows if r["label"] in MAP]
    dropped = [r for r in rows if r["label"] not in MAP]
    print(f"Total CSV: {len(rows)}, dipakai: {len(kept)}, "
          f"dibuang (unreadable/manual_review): {len(dropped)}")
    for r in kept:
        if r["label"] == "plat-jogja":
            assert r["prefix"].strip().upper() == "AB", \
                f"Label jogja tapi prefix bukan AB: {r}"
        elif r["prefix"].strip().upper() == "AB":
            raise AssertionError(f"Prefix AB tapi label bukan jogja: {r}")

    if args.reshuffle:
        rng = random.Random(args.seed)
        by_class = {}
        for r in kept:
            by_class.setdefault(r["label"], []).append(r)
        splits = {"train": [], "valid": [], "test": []}
        for label, items in by_class.items():
            rng.shuffle(items)
            n = len(items)
            n_test, n_val = max(1, int(n * 0.1)), max(1, int(n * 0.1))
            splits["test"] += items[:n_test]
            splits["valid"] += items[n_test:n_test + n_val]
            splits["train"] += items[n_test + n_val:]
    else:
        splits = {"train": [], "valid": [], "test": []}
        for r in kept:
            splits[r["split"]].append(r)

    src = Path(args.src)
    dst = Path(args.dst)
    for split, items in splits.items():
        for r in items:
            img = src / split / "images" / r["image"]
            if not img.exists():
                print("Tidak ditemukan:", r["image"])
                continue
            label_src = src / split / "labels" / (img.stem + ".txt")
            out_img = dst / split / "images" / img.name
            out_lbl = dst / split / "labels" / (img.stem + ".txt")
            out_img.parent.mkdir(parents=True, exist_ok=True)
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img, out_img)
            new_lines = []
            for ln in label_src.read_text().splitlines():
                if not ln.strip():
                    continue
                parts = ln.split()
                w, h = float(parts[3]), float(parts[4])
                # Buang kotak degeneratif (mis. 1-2 px) yang hanya noise
                # anotasi, bukan plat kedua yang nyata.
                if w < 0.01 or h < 0.01:
                    print(f"  kotak degeneratif dibuang: {img.name} "
                          f"w={w} h={h}")
                    continue
                new_lines.append(f"{MAP[r['label']]} " +
                                 " ".join(parts[1:]))
            out_lbl.write_text("\n".join(new_lines) + "\n")
        print(split, len(items))
    (dst / "data.yaml").write_text(
        "path: .\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 2\n"
        "names:\n"
        "  0: plat-jogja\n"
        "  1: plat-non-jogja\n")
    print("data.yaml ditulis di", dst / "data.yaml")


if __name__ == "__main__":
    main()
