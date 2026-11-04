"""Evaluasi deteksi 2 kelas Jogja/non-Jogja di split test/valid.

Berbeda dengan `yolo val` (mAP saja), skrip ini membangun confusion
matrix klasifikasi dari kotak yang berhasil dipasangkan:
  - pasangkan prediksi ke ground truth dengan IoU >= 0.5 (greedy,
    confidence tertinggi dulu),
  - kotak GT tanpa pasangan = missed,
  - prediksi tanpa pasangan = false positive (bukan plat / salah lokasi),
  - pasangan dengan kelas berbeda = salah klasifikasi Jogja/non-Jogja.

Keluaran: CSV per gambar + ringkasan di stdout + confusion matrix.

Penggunaan:
    .venv/bin/python scripts/evaluate_detection.py \
        --weights runs/yolo11_jogja/weights/best.pt \
        --split test --out runs/eval_test.csv
"""
import argparse
import csv
from pathlib import Path

import cv2


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter) if area_a + area_b - inter else 0


def xywhn_to_xyxy(box, w, h):
    cx, cy, bw, bh = box
    return (cx - bw / 2) * w, (cy - bh / 2) * h, \
        (cx + bw / 2) * w, (cy + bh / 2) * h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--root", default="data/jogja_2cls")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--out", default="runs/eval_test.csv")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = {0: "plat-jogja", 1: "plat-non-jogja"}
    rows, tp = [], {"jogja": 0, "non": 0}
    wrong_cls, missed, fp = 0, 0, 0
    img_dir = Path(args.root) / args.split / "images"
    lbl_dir = Path(args.root) / args.split / "labels"
    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        im = cv2.imread(str(img_path))
        h, w = im.shape[:2]
        gt = []
        lbl = lbl_dir / (img_path.stem + ".txt")
        if lbl.exists():
            for ln in lbl.read_text().splitlines():
                if ln.strip():
                    p = ln.split()
                    gt.append((int(p[0]), xywhn_to_xyxy(
                        [float(x) for x in p[1:5]], w, h)))
        res = model.predict(str(img_path), imgsz=args.imgsz,
                            conf=args.conf, verbose=False)[0]
        preds = []
        if res.boxes is not None:
            for box, cls, conf in zip(res.boxes.xyxy, res.boxes.cls,
                                      res.boxes.conf):
                preds.append((int(cls), tuple(float(x) for x in box),
                              float(conf)))
        preds.sort(key=lambda x: -x[2])
        used = set()
        for gi, (gcls, gbox) in enumerate(gt):
            best, best_iou = -1, 0.5
            for pi, (pcls, pbox, _) in enumerate(preds):
                if pi in used:
                    continue
                v = iou(gbox, pbox)
                if v > best_iou:
                    best, best_iou = pi, v
            if best >= 0:
                used.add(best)
                pcls = preds[best][0]
                if pcls == gcls:
                    tp["jogja" if gcls == 0 else "non"] += 1
                    rows.append({"image": img_path.name,
                                 "gt": names[gcls], "pred": names[pcls],
                                 "status": "benar"})
                else:
                    wrong_cls += 1
                    rows.append({"image": img_path.name,
                                 "gt": names[gcls], "pred": names[pcls],
                                 "status": "salah_kelas"})
            else:
                missed += 1
                rows.append({"image": img_path.name, "gt": names[gcls],
                             "pred": "", "status": "terlewat"})
        for pi, (pcls, _, _) in enumerate(preds):
            if pi not in used:
                fp += 1
                rows.append({"image": img_path.name, "gt": "",
                             "pred": names[pcls], "status": "false_positive"})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["image", "gt", "pred",
                                            "status"])
        wtr.writeheader()
        wtr.writerows(rows)
    total_gt = tp["jogja"] + tp["non"] + wrong_cls + missed
    print(f"GT total: {total_gt}, benar Jogja: {tp['jogja']}, "
          f"benar non-Jogja: {tp['non']}, salah kelas: {wrong_cls}, "
          f"terlewat: {missed}, false positive: {fp}")
    print(f"Detail per gambar: {out}")


if __name__ == "__main__":
    main()
