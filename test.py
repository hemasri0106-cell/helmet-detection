from ultralytics import YOLO

# Load trained model
model = YOLO(r"C:\Users\Hemasri\OneDrive\Desktop\helmet detection\runs\detect\train2\weights\best.pt")

# Run prediction
results = model.predict(
    source="image10.jpg",
    imgsz=416,
    conf=0.4,
    save=True,
    project="output",     # force save folder
    name="helmet_test",   # subfolder name
    exist_ok=True
)

print("Detection complete. Check the 'output/helmet_test' folder.")
