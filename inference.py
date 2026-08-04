import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import config
import uuid
import numpy as np

# Load model globally
try:
    model = YOLO(config.MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: Could not load model from {config.MODEL_PATH}. Error: {e}")

def get_prediction_summary(results):
    with_helmet_count = 0
    without_helmet_count = 0
    total_conf = 0.0
    total_detections = 0
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls_id == 0:
                with_helmet_count += 1
            elif cls_id == 1:
                without_helmet_count += 1
                
            total_conf += conf
            total_detections += 1
            
    avg_conf = (total_conf / total_detections) if total_detections > 0 else 0.0
    
    return {
        "With Helmet": with_helmet_count,
        "Without Helmet": without_helmet_count,
        "Average Confidence": f"{avg_conf:.2f}"
    }

def predict_image(image_path):
    if model is None:
        return None, {"Error": "Model not loaded"}
    
    print("\n--- INFERENCE DEBUG LOG (IMAGE) ---")
    print(f"Loaded model path: {config.MODEL_PATH}")
    print(f"Model names: {model.names}")
    print(f"Input image path: {image_path}")
    print("Parameters: imgsz=416, conf=0.4")
    
    # Exactly matches standalone test.py script parameters
    start_time = time.time()
    results = model.predict(source=str(image_path), imgsz=416, conf=0.4, save=False, verbose=False)
    inference_time = time.time() - start_time
    
    r = results[0]
    print(f"Predicted class IDs: {[int(c) for c in r.boxes.cls]}")
    print(f"Confidence values: {[float(c) for c in r.boxes.conf]}")
    print(f"Bounding box coordinates: {[box.tolist() for box in r.boxes.xyxy]}")
    print(f"Inference time: {inference_time:.4f}s")
    
    # YOLO returns a BGR image array from plot()
    annotated_img_bgr = r.plot()
    
    # Convert strictly for Gradio gr.Image rendering (Gradio expects RGB)
    annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
        
    summary = get_prediction_summary(results)
    summary["Inference Time"] = f"{inference_time:.2f}s"
    
    print("-----------------------------------\n")
    return annotated_img_rgb, summary

def predict_video(video_path):
    if model is None:
        return None, {"Error": "Model not loaded"}
        
    print("\n--- INFERENCE DEBUG LOG (VIDEO) ---")
    print(f"Loaded model path: {config.MODEL_PATH}")
    print(f"Model names: {model.names}")
    print(f"Video path: {video_path}")
    print("Parameters: model(frame) [Default Confidence/Size]")
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, {"Error": "Could not open video"}
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if fps == 0:
        fps = 25
        
    output_filename = f"out_{uuid.uuid4().hex[:8]}.mp4"
    output_path = config.CACHE_DIR / output_filename
    
    print(f"Video codec parameters: Resolution {width}x{height}, FPS {fps}")
    print(f"Output video path: {output_path}")
    
    # Using 'mp4v' for MP4 saving
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frames_processed = 0
    total_with = 0
    total_without = 0
    total_conf_sum = 0.0
    total_detections = 0
    
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frames_processed += 1
        
        # Exactly matches standalone video_detection.py script parameters
        results = model(frame, verbose=False)
        annotated_frame_bgr = results[0].plot()
        
        # Write directly to VideoWriter (expects BGR)
        out.write(annotated_frame_bgr)
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id == 0:
                    total_with += 1
                elif cls_id == 1:
                    total_without += 1
                    
                total_conf_sum += conf
                total_detections += 1
                
    cap.release()
    out.release()
    
    inference_time = time.time() - start_time
    avg_conf = (total_conf_sum / total_detections) if total_detections > 0 else 0.0
    
    print(f"Frames processed: {frames_processed}")
    print(f"Inference time: {inference_time:.4f}s")
    print("-----------------------------------\n")
    
    summary = {
        "Frames Processed": frames_processed,
        "Helmet Detections": total_with,
        "No Helmet Detections": total_without,
        "Average Confidence": f"{avg_conf:.2f}",
        "Inference Time": f"{inference_time:.1f}s"
    }
    
    return str(output_path), summary
