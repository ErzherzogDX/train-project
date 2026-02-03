import argparse
import json
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve(root: Path, p):
    if p is None:
        return None
    if isinstance(p, (list, tuple)):
        return [_resolve(root, x) for x in p]
    pp = Path(str(p))
    if pp.is_absolute():
        return pp
    return (root / pp).resolve()


def load_images_from_data_yaml(data_yaml: str, split_key: str, limit: int):
    p = Path(data_yaml).resolve()
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    root = p.parent
    split = _resolve(root, cfg.get(split_key))
    roots = split if isinstance(split, list) else [split]
    imgs = []
    for r in roots:
        if r is None or not Path(r).exists():
            continue
        for x in Path(r).rglob("*"):
            if x.is_file() and x.suffix.lower() in IMG_EXTS:
                imgs.append(x)
                if len(imgs) >= limit:
                    return imgs
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", type=str, default="benchmark.json")
    args = ap.parse_args()

    imgs = load_images_from_data_yaml(args.data, args.split, args.limit)
    if len(imgs) == 0:
        raise RuntimeError("no images found for benchmarking")

    model = YOLO(args.weights)

    for i in range(min(args.warmup, len(imgs))):
        model.predict(source=str(imgs[i]), imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)

    t0 = time.perf_counter()
    for p in imgs:
        model.predict(source=str(p), imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)
    t1 = time.perf_counter()

    total_s = t1 - t0
    ms_img = total_s * 1000.0 / len(imgs)
    fps = len(imgs) / total_s if total_s > 0 else float("inf")

    out = {"n": len(imgs), "total_s": float(total_s), "ms_per_image": float(ms_img), "fps": float(fps)}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
