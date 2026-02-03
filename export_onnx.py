import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--outdir", type=str, default="exports")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    res = model.export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True)
    print(res)


if __name__ == "__main__":
    main()
