from ultralytics import YOLO

# Load the YOLOv8n or YOLOv11 model architecture
model = YOLO("yolov8n.pt")  # or your own path like "yolo11s.pt"

# Train
model.train(
    data="configs/data.yaml",  # path to data.yaml
    epochs=100,
    batch=8,
    imgsz=640
)
