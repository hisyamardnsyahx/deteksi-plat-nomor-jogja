"""Training YOLOv11 (fine-tune / transfer learning).

Contoh baseline 1 kelas:
    python scripts/train.py --data configs/data_roboflow_1cls.yaml \
        --model yolo11n.pt --epochs 100 --imgsz 640 --batch 16
Contoh 2 kelas Jogja (imgsz besar, tanpa flip):
    python scripts/train.py --data configs/data_jogja_2cls.yaml \
        --model yolo11n.pt --epochs 100 --imgsz 960 --batch 8 --fliplr 0.0
Catatan: fliplr=0.0 itu sengaja untuk tugas Jogja (lihat README).
Jangan ubah jadi 0.5 tanpa alasan tertulis di laporan.
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fliplr", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="yolo11_plat")
    ap.add_argument("--workers", type=int, default=2,
                    help="2 untuk CPU 4-core VM ini; naikkan di GPU")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, patience=args.patience, seed=args.seed,
                workers=args.workers,
                fliplr=args.fliplr, flipud=0.0, mosaic=1.0, close_mosaic=10,
                project=args.project, name=args.name)
    print("Selesai. Evaluasi standar (jangan pakai conf deployment "
          "untuk headline mAP):")
    print(f"  yolo val model={args.project}/{args.name}/weights/best.pt "
          f"data={args.data} imgsz={args.imgsz}")


if __name__ == "__main__":
    main()
