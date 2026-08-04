import cv2
import time
from ultralytics import YOLO
import config
from inference import predict_image, predict_video
import numpy as np
import os

print("--- STEP 2: VERIFY MODEL ---")
print(f"Loaded model path: {config.MODEL_PATH}")
model = YOLO(config.MODEL_PATH)
print(f"Model names: {model.names}")

print("\n--- STEP 3: VERIFY INFERENCE CONSISTENCY (IMAGE) ---")
test_img = "test.jpg"

# Standalone test.py simulation
print("\n[Standalone simulation - test.py]")
results_standalone = model.predict(source=test_img, imgsz=416, conf=0.4, save=False, verbose=False)
r_stand = results_standalone[0]
stand_boxes = r_stand.boxes
print(f"Predicted class IDs: {[int(c) for c in stand_boxes.cls]}")
print(f"Confidence scores: {[float(c) for c in stand_boxes.conf]}")
print(f"Bounding box coordinates: {[box.xyxy[0].tolist() for box in stand_boxes]}")
print(f"Number of detections: {len(stand_boxes)}")

# Gradio inference.py simulation
print("\n[Gradio simulation - predict_image()]")
# predict_image returns annotated_img_rgb, summary
out_img, summary = predict_image(test_img)
print(f"Summary returned: {summary}")
# We can't directly get boxes from predict_image because it only returns the image and summary.
# But we can see if the summary matches the standalone boxes.
print(f"Number of detections (from summary): {summary.get('With Helmet', 0) + summary.get('Without Helmet', 0)}")

# Let's check what results inference.py ACTUALLY generates internally
results_inference = model.predict(source=str(test_img), imgsz=416, conf=0.4, save=False, verbose=False)
r_inf = results_inference[0]
inf_boxes = r_inf.boxes
print(f"\nInternal inference.py logic:")
print(f"Predicted class IDs: {[int(c) for c in inf_boxes.cls]}")
print(f"Confidence scores: {[float(c) for c in inf_boxes.conf]}")
print(f"Bounding box coordinates: {[box.xyxy[0].tolist() for box in inf_boxes]}")

print("\n--- STEP 6: VERIFY INFERENCE CONSISTENCY (VIDEO) ---")
# Standalone video_detection.py simulation
print("\n[Standalone simulation - video_detection.py]")
print("Code uses: results = model(frame) # reset to default confidence threshold")

# Gradio predict_video simulation
print("\n[Gradio simulation - predict_video()]")
print("Code uses: results = model.predict(source=frame, imgsz=416, conf=0.4, save=False, verbose=False)")

print("\nConclusion on differences:")
print("1. Image prediction logic is identical in code (model.predict(source=..., imgsz=416, conf=0.4)). Any difference might be due to caching or the RGB conversion.")
print("2. Video prediction logic is DIFFERENT. Standalone uses model(frame), Gradio uses model.predict(..., imgsz=416, conf=0.4).")
