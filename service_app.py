import hashlib
import io
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from PIL import Image
from ultralytics import YOLO


def sha256_file(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


WEIGHTS = Path(os.environ.get("WEIGHTS", "artifacts/best.pt")).resolve()
DEFAULT_CONF = float(os.environ.get("CONF", "0.25"))
DEFAULT_IOU = float(os.environ.get("IOU", "0.7"))
DEFAULT_IMGSZ = int(os.environ.get("IMGSZ", "640"))
DEVICE = os.environ.get("DEVICE", "")

MODEL = YOLO(str(WEIGHTS))
WEIGHTS_SHA256 = sha256_file(WEIGHTS) if WEIGHTS.exists() else None

APP = FastAPI(title="Railcars YOLO inference")


@APP.get("/health")
def health():
    names = None
    try:
        names = getattr(MODEL.model, "names", None)
    except Exception:
        names = None
    return {
        "status": "ok",
        "weights": str(WEIGHTS),
        "weights_sha256": WEIGHTS_SHA256,
        "default_conf": DEFAULT_CONF,
        "default_iou": DEFAULT_IOU,
        "default_imgsz": DEFAULT_IMGSZ,
        "device": DEVICE,
        "names": names,
    }


def _predict_array(arr: np.ndarray, conf: float, iou: float, imgsz: int, max_det: int | None):
    t0 = time.perf_counter()
    res = MODEL.predict(source=arr, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=DEVICE, verbose=False)[0]
    t1 = time.perf_counter()

    boxes = []
    for b in res.boxes:
        cls = int(b.cls.item())
        cf = float(b.conf.item())
        xyxy = [float(x) for x in b.xyxy.squeeze(0).tolist()]
        boxes.append({"cls": cls, "label": res.names.get(cls, str(cls)), "conf": cf, "xyxy": xyxy})

    h, w = arr.shape[:2]
    return {"n": len(boxes), "boxes": boxes, "inference_ms": float((t1 - t0) * 1000.0), "image_size": {"w": int(w), "h": int(h)}}


@APP.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(DEFAULT_CONF),
    iou: float = Query(DEFAULT_IOU),
    imgsz: int = Query(DEFAULT_IMGSZ),
    max_det: int | None = Query(None),
) -> dict[str, Any]:
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    arr = np.array(img)
    out = _predict_array(arr, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det)
    out["filename"] = file.filename
    return out


@APP.post("/predict_batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    conf: float = Query(DEFAULT_CONF),
    iou: float = Query(DEFAULT_IOU),
    imgsz: int = Query(DEFAULT_IMGSZ),
    max_det: int | None = Query(None),
) -> dict[str, Any]:
    outputs = []
    for f in files:
        content = await f.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        arr = np.array(img)
        out = _predict_array(arr, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det)
        out["filename"] = f.filename
        outputs.append(out)
    return {"n_files": len(outputs), "results": outputs}
