import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--outdir", type=str, default="predictions")
    args = ap.parse_args()

    model = YOLO(args.weights)
    results = model.predict(source=args.image, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device, verbose=False)

    r = results[0]
    boxes = []
    for b in r.boxes:
        cls = int(b.cls.item())
        conf = float(b.conf.item())
        xyxy = [float(x) for x in b.xyxy.squeeze(0).tolist()]
        boxes.append({"cls": cls, "label": r.names.get(cls, str(cls)), "conf": conf, "xyxy": xyxy})

    out = {"image": args.image, "n": len(boxes), "boxes": boxes}
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.save:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        save_path = str(Path(args.outdir) / (Path(args.image).stem + "_pred.jpg"))
        r.save(filename=save_path)
        print(save_path)


if __name__ == "__main__":
    main()
