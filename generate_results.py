from ultralytics import YOLO
import os
import csv

# Load model
model = YOLO(r"C:\Users\Hemasri\OneDrive\Desktop\helmet detection\runs\detect\train2\weights\best.pt")

# Folder of images to test
IMAGE_FOLDER = "dataset/images/val"

# Output CSV
csv_file = "results.csv"

with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image", "class", "confidence"])

    for img in os.listdir(IMAGE_FOLDER):
        img_path = os.path.join(IMAGE_FOLDER, img)

        results = model.predict(source=img_path, conf=0.4)

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])

                    writer.writerow([img, class_name, confidence])

print("CSV file created!")