# Railcars YOLOv8: detection + inference service

## Install
```bash
pip install -r requirements.txt
```

## Download dataset (Colab/Kaggle API configured)
```bash
kaggle datasets download -d avoronin/railcars-dataset-for-yolo -p data_raw --unzip
```

## Dataset checks (labels/images stats + visual sanity-check)
```bash
python -c "from data_checks import load_data_yaml, dataset_stats; cfg=load_data_yaml('data_raw/data.yaml'); print(cfg); print(dataset_stats(cfg['train'])); print(dataset_stats(cfg['val']))"
```

## Train (config-driven)
```bash
python train_yolo.py --config config_baseline.yaml
python train_yolo.py --config config_improved.yaml
```

Outputs are written to `artifacts/<run_tag>/`:
- `best.pt`, `last.pt`
- `metrics.json`
- `meta.json`
- `run_config.json`

## Benchmark inference latency
```bash
python benchmark_yolo.py --weights artifacts/<run_tag>/best.pt --data data_raw/data.yaml --split val --limit 64
```

## Export ONNX
```bash
python export_onnx.py --weights artifacts/<run_tag>/best.pt --imgsz 640
```

## CLI inference
```bash
python infer_yolo.py --weights artifacts/<run_tag>/best.pt --image path/to/image.jpg --save
```

## FastAPI service
```bash
WEIGHTS=artifacts/<run_tag>/best.pt uvicorn service_app:APP --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /predict?conf=0.25&iou=0.7&imgsz=640`
- `POST /predict_batch?conf=0.25&iou=0.7&imgsz=640`
