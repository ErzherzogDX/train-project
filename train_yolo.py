import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--model", type=str, default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--project", type=str, default="runs")
    ap.add_argument("--name", type=str, default="train")
    ap.add_argument("--artifacts", type=str, default="artifacts")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        verbose=False,
    )

    run_dir = Path(model.trainer.save_dir)
    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    if best_pt.exists():
        (artifacts / "best.pt").write_bytes(best_pt.read_bytes())
    if last_pt.exists():
        (artifacts / "last.pt").write_bytes(last_pt.read_bytes())

    try:
        names = model.model.names
    except Exception:
        names = None

    meta = {
        "data": args.data,
        "base_model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "run_dir": str(run_dir),
        "names": names,
    }
    (artifacts / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(str(run_dir))
    print(str(artifacts / "best.pt"))


if __name__ == "__main__":
    main()
