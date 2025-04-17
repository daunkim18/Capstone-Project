import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='configs/yolo_v11.yaml')  # Your model config
parser.add_argument('--data', default='configs/data.yaml')     # Your dataset config
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--imgsz', type=int, default=640)
args = parser.parse_args()

model = YOLO('yolov8n.pt')  # Or yolov8s.pt, yolov8m.pt etc.
  # Load model architecture
model.train(
    data=args.data,
    epochs=args.epochs,
    batch=args.batch,
    imgsz=args.imgsz
)
