import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO


def sha256_file(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _device_arg(device):
    if device in ("auto", "", None):
        return ""
    return str(device)


def _now_tag():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _dump_json(p: Path, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--artifacts", type=str, default="artifacts")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_tag = str(cfg.get("name", "run")) + "_" + _now_tag()

    artifacts_root = Path(args.artifacts).resolve()
    artifacts_dir = artifacts_root / run_tag
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    _dump_json(artifacts_dir / "run_config.json", cfg)

    model = YOLO(cfg["base_model"])
    model.train(
        data=cfg["data_yaml"],
        epochs=int(cfg.get("epochs", 10)),
        imgsz=int(cfg.get("imgsz", 640)),
        batch=int(cfg.get("batch", 16)),
        device=_device_arg(cfg.get("device", "auto")),
        seed=int(cfg.get("seed", 42)),
        project=str(artifacts_dir / "runs"),
        name="train",
        verbose=False,
    )

    run_dir = Path(model.trainer.save_dir)
    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    if best_pt.exists():
        (artifacts_dir / "best.pt").write_bytes(best_pt.read_bytes())
    if last_pt.exists():
        (artifacts_dir / "last.pt").write_bytes(last_pt.read_bytes())

    metrics = {}
    try:
        val_res = model.val(
            data=cfg["data_yaml"],
            imgsz=int(cfg.get("imgsz", 640)),
            device=_device_arg(cfg.get("device", "auto")),
            verbose=False,
        )
        try:
            metrics = {"map50": float(val_res.box.map50), "map": float(val_res.box.map)}
        except Exception:
            metrics = {"val": str(val_res)}
    except Exception as e:
        metrics = {"val_error": str(e)}

    _dump_json(artifacts_dir / "metrics.json", metrics)

    meta = {
        "run_tag": run_tag,
        "data_yaml": cfg["data_yaml"],
        "base_model": cfg["base_model"],
        "train_run_dir": str(run_dir),
        "weights_sha256": sha256_file(artifacts_dir / "best.pt") if (artifacts_dir / "best.pt").exists() else None,
        "names": getattr(model.model, "names", None),
        "metrics": metrics,
    }
    _dump_json(artifacts_dir / "meta.json", meta)

    print(str(artifacts_dir))


if __name__ == "__main__":
    main()
