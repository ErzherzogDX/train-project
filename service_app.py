import io
import os
from typing import Any

import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO


WEIGHTS = os.environ.get("WEIGHTS", "artifacts/best.pt")
CONF = float(os.environ.get("CONF", "0.25"))
IOU = float(os.environ.get("IOU", "0.7"))

MODEL = YOLO(WEIGHTS)
APP = FastAPI(title="Railcars YOLO inference")


@APP.get("/health")
def health():
    return {"status": "ok", "weights": WEIGHTS, "conf": CONF, "iou": IOU}


@APP.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    arr = np.array(img)
    res = MODEL.predict(source=arr, conf=CONF, iou=IOU, verbose=False)[0]

    boxes = []
    for b in res.boxes:
        cls = int(b.cls.item())
        conf = float(b.conf.item())
        xyxy = [float(x) for x in b.xyxy.squeeze(0).tolist()]
        boxes.append({"cls": cls, "label": res.names.get(cls, str(cls)), "conf": conf, "xyxy": xyxy})

    return {"n": len(boxes), "boxes": boxes, "names": res.names}
