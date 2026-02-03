import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--out", type=str, default="metrics.json")
    args = ap.parse_args()

    model = YOLO(args.weights)
    res = model.val(data=args.data, imgsz=args.imgsz, device=args.device, verbose=False)
    out = {}
    try:
        out = {"map50": float(res.box.map50), "map": float(res.box.map)}
    except Exception:
        out = {"val": str(res)}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
