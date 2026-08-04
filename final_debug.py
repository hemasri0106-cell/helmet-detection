import cv2
import time
from ultralytics import YOLO
import config
from inference import predict_image, predict_video
import numpy as np

print("--- STEP 2: VERIFY MODEL ---")
model = YOLO(config.MODEL_PATH)
print(f"Loaded model: {config.MODEL_PATH}")
print(f"Model names: {model.names}")

print("\n--- STEP 3: VERIFY INFERENCE CONSISTENCY ---")
test_img = "test.jpg"

print("\n[Standalone simulation - test.py on test.jpg]")
results_standalone = model.predict(source=test_img, imgsz=416, conf=0.4, save=False, verbose=False)
r_stand = results_standalone[0]
stand_boxes = r_stand.boxes
print(f"Predicted class IDs: {[int(c) for c in stand_boxes.cls]}")
print(f"Confidence scores: {[float(c) for c in stand_boxes.conf]}")
print(f"Bounding box coordinates: {[box.xyxy[0].tolist() for box in stand_boxes]}")
print(f"Number of detections: {len(stand_boxes)}")

print("\n[Gradio simulation - predict_image() on test.jpg]")
out_img, summary = predict_image(test_img)
print(f"Summary returned: {summary}")

# Also test image10.jpg to see the discrepancy
print("\n[Standalone simulation - test.py on image10.jpg]")
results_standalone_10 = model.predict(source="image10.jpg", imgsz=416, conf=0.4, save=False, verbose=False)
r_stand_10 = results_standalone_10[0]
stand_boxes_10 = r_stand_10.boxes
print(f"Predicted class IDs: {[int(c) for c in stand_boxes_10.cls]}")
print(f"Confidence scores: {[float(c) for c in stand_boxes_10.conf]}")
print(f"Bounding box coordinates: {[box.xyxy[0].tolist() for box in stand_boxes_10]}")
print(f"Number of detections: {len(stand_boxes_10)}")

print("\n[Gradio simulation - predict_image() on image10.jpg]")
out_img_10, summary_10 = predict_image("image10.jpg")
print(f"Summary returned: {summary_10}")

print("\n--- STEP 6: VERIFY INFERENCE CONSISTENCY (VIDEO) ---")
print("Comparing video_detection.py and predict_video()")
print("video_detection.py logic: results = model(frame)")
print("inference.py logic: results = model.predict(source=frame, imgsz=416, conf=0.4, save=False, verbose=False)")
