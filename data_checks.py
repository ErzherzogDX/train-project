import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

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


def load_data_yaml(data_yaml: str):
    p = Path(data_yaml).resolve()
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    root = p.parent
    train = _resolve(root, cfg.get("train"))
    val = _resolve(root, cfg.get("val"))
    test = _resolve(root, cfg.get("test"))
    names = cfg.get("names")
    nc = cfg.get("nc")
    return {"path": str(p), "root": str(root), "train": train, "val": val, "test": test, "names": names, "nc": nc}


def iter_images(images_root: Path):
    if images_root is None or not images_root.exists():
        return
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def yolo_label_path(img_path: Path):
    parts = list(img_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return img_path.with_suffix(".txt")


def parse_yolo_labels(lbl_path: Path):
    if not lbl_path.exists():
        return []
    lines = [x.strip() for x in lbl_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    out = []
    for ln in lines:
        toks = ln.split()
        if len(toks) < 5:
            continue
        cls = int(float(toks[0]))
        x, y, w, h = map(float, toks[1:5])
        out.append((cls, x, y, w, h))
    return out


def dataset_stats(split_root: Path, max_images: int | None = None, seed: int = 42):
    imgs = list(iter_images(split_root))
    if max_images is not None:
        rnd = random.Random(seed)
        rnd.shuffle(imgs)
        imgs = imgs[:max_images]
    total = len(imgs)
    missing_labels = 0
    empty_labels = 0
    boxes_total = 0
    cls_counts = {}
    for img in imgs:
        lbl = yolo_label_path(img)
        if not lbl.exists():
            missing_labels += 1
            continue
        boxes = parse_yolo_labels(lbl)
        if len(boxes) == 0:
            empty_labels += 1
        boxes_total += len(boxes)
        for c, *_ in boxes:
            cls_counts[c] = cls_counts.get(c, 0) + 1
    return {
        "images": total,
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "boxes_total": boxes_total,
        "cls_counts": dict(sorted(cls_counts.items(), key=lambda x: (-x[1], x[0]))),
    }


def draw_boxes(img_path: Path, names=None):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    boxes = parse_yolo_labels(yolo_label_path(img_path))
    dr = ImageDraw.Draw(img)
    for cls, xc, yc, bw, bh in boxes:
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h
        dr.rectangle([x1, y1, x2, y2], outline="red", width=2)
        if names and isinstance(names, dict) and cls in names:
            dr.text((x1, y1), str(names[cls]), fill="red")
    return img
