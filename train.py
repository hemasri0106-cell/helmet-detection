from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=10,      # keep small for first test
    imgsz=416,      # smaller image size
    batch=4,        # very safe batch size for 8GB RAM
    device="cpu"    # force CPU
)
