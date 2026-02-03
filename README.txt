Railcars YOLO project

1) Install:
   pip install -r requirements.txt

2) Download dataset (Colab/Kaggle API configured):
   kaggle datasets download -d avoronin/railcars-dataset-for-yolo -p data_raw --unzip

3) Train:
   python train_yolo.py --data data_raw/data.yaml --epochs 10 --imgsz 640 --batch 16

4) Inference (CLI):
   python infer_yolo.py --weights artifacts/best.pt --image path/to/image.jpg --save

5) Service:
   uvicorn service_app:APP --host 0.0.0.0 --port 8000
   POST /predict with multipart form field 'file'
