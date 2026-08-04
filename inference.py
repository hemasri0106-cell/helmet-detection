import cv2
import time
import os
import psutil
import traceback
from pathlib import Path
from ultralytics import YOLO
import config
import uuid
import numpy as np

# Utility to check RAM
def get_memory_mb():
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0

# Cleanup utility for cache
def cleanup_old_files():
    try:
        now = time.time()
        count = 0
        for f in config.CACHE_DIR.glob("*"):
            if f.is_file() and (now - f.stat().st_mtime) > 900:  # 15 minutes
                f.unlink()
                count += 1
        if count > 0:
            print(f"🧹 Cleaned up {count} old temporary files from cache.")
    except Exception as e:
        print(f"⚠️ Warning: Cache cleanup failed. {e}")

print(f"\n[INIT] Starting Backend Initialization. Process ID: {os.getpid()}")
print(f"[INIT] Initial Memory: {get_memory_mb():.2f} MB")

# Load model globally
start_load = time.time()
try:
    model = YOLO(config.MODEL_PATH)
    load_time = time.time() - start_load
    print(f"[INIT] Model loaded successfully in {load_time:.2f}s. Memory: {get_memory_mb():.2f} MB")
except Exception as e:
    model = None
    print(f"[INIT-ERROR] Could not load model from {config.MODEL_PATH}. Error: {e}")
    traceback.print_exc()



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
    print("\n=============================================")
    print("      IMAGE INFERENCE REQUEST RECEIVED       ")
    print("=============================================")
    req_start_time = time.time()
    
    try:
        cleanup_old_files()
        
        if model is None:
            return None, {"Error": "Model not loaded globally"}
            
        mem_start = get_memory_mb()
        print(f"[{time.time()-req_start_time:.3f}s] Memory Usage Start: {mem_start:.2f} MB")
        
        # 1. Image Loading & Sizing
        load_start = time.time()
        file_size = os.path.getsize(image_path) / 1024 / 1024
        print(f"[{time.time()-req_start_time:.3f}s] File Size: {file_size:.2f} MB")
        
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError("cv2.imread failed to load the image.")
            
        h, w = img.shape[:2]
        print(f"[{time.time()-req_start_time:.3f}s] Original Dimensions: {w}x{h}")
        
        # Resize if too large
        if max(w, h) > 1280:
            scale = 1280 / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            print(f"[{time.time()-req_start_time:.3f}s] ⚠️ Image too large. Resizing to: {new_w}x{new_h}")
            img = cv2.resize(img, (new_w, new_h))
            
        print(f"[{time.time()-req_start_time:.3f}s] Image Load/Resize Time: {time.time()-load_start:.3f}s")

        # 2. YOLO Prediction
        print(f"[{time.time()-req_start_time:.3f}s] Starting YOLO model.predict...")
        infer_start = time.time()
        results = model.predict(source=img, imgsz=416, conf=0.4, save=False, verbose=False)
        infer_time = time.time() - infer_start
        print(f"[{time.time()-req_start_time:.3f}s] ✅ YOLO Inference Time: {infer_time:.4f}s")
        
        # 3. Bounding Box Rendering
        plot_start = time.time()
        r = results[0]
        annotated_img_bgr = r.plot()
        plot_time = time.time() - plot_start
        print(f"[{time.time()-req_start_time:.3f}s] Bounding Box Render Time: {plot_time:.4f}s")
        
        # 4. RGB Conversion
        rgb_start = time.time()
        annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
        rgb_time = time.time() - rgb_start
        print(f"[{time.time()-req_start_time:.3f}s] RGB Conversion Time: {rgb_time:.4f}s")
        
        # 5. Summary Generation
        summary = get_prediction_summary(results)
        summary["Inference Time"] = f"{infer_time:.2f}s"
        
        mem_end = get_memory_mb()
        total_time = time.time() - req_start_time
        print(f"[{total_time:.3f}s] Memory Usage End: {mem_end:.2f} MB (Delta: {mem_end - mem_start:+.2f} MB)")
        print("=============================================\n")
        
        return annotated_img_rgb, summary

    except Exception as e:
        print("\n❌ CRITICAL ERROR IN PREDICT_IMAGE:")
        traceback.print_exc()
        print("=============================================\n")
        return None, {"Error": f"Backend failure: {str(e)}"}

def predict_video(video_path):
    print("\n=============================================")
    print("      VIDEO INFERENCE REQUEST RECEIVED       ")
    print("=============================================")
    req_start_time = time.time()
    
    try:
        cleanup_old_files()
        
        if model is None:
            return None, {"Error": "Model not loaded globally"}
            
        mem_start = get_memory_mb()
        print(f"[{time.time()-req_start_time:.3f}s] Memory Usage Start: {mem_start:.2f} MB")
        
        file_size = os.path.getsize(video_path) / 1024 / 1024
        print(f"[{time.time()-req_start_time:.3f}s] Video File Size: {file_size:.2f} MB")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Could not open video file.")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0: fps = 25
        
        print(f"[{time.time()-req_start_time:.3f}s] Original Video: {width}x{height} @ {fps}fps ({total_frames} frames)")
        
        # Check resizing
        target_w, target_h = width, height
        if max(width, height) > 1280:
            scale = 1280 / max(width, height)
            target_w, target_h = int(width * scale), int(height * scale)
            print(f"[{time.time()-req_start_time:.3f}s] ⚠️ Video too large. Resizing frames to: {target_w}x{target_h}")
            
        output_filename = f"out_{uuid.uuid4().hex[:8]}.mp4"
        output_path = config.CACHE_DIR / output_filename
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (target_w, target_h))
        
        frames_processed = 0
        total_with, total_without = 0, 0
        total_conf_sum = 0.0
        total_detections = 0
        
        print(f"[{time.time()-req_start_time:.3f}s] Starting frame processing loop...")
        loop_start = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if target_w != width or target_h != height:
                frame = cv2.resize(frame, (target_w, target_h))
                
            frames_processed += 1
            
            results = model(frame, verbose=False)
            annotated_frame_bgr = results[0].plot()
            out.write(annotated_frame_bgr)
            
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 0: total_with += 1
                    elif cls_id == 1: total_without += 1
                    total_conf_sum += float(box.conf[0])
                    total_detections += 1
                    
        cap.release()
        out.release()
        
        loop_time = time.time() - loop_start
        print(f"[{time.time()-req_start_time:.3f}s] ✅ Frame Loop Finished. Time: {loop_time:.2f}s")
        print(f"[{time.time()-req_start_time:.3f}s] Processed {frames_processed}/{total_frames} frames.")
        
        avg_conf = (total_conf_sum / total_detections) if total_detections > 0 else 0.0
        summary = {
            "Frames Processed": frames_processed,
            "Helmet Detections": total_with,
            "No Helmet Detections": total_without,
            "Average Confidence": f"{avg_conf:.2f}",
            "Inference Time": f"{loop_time:.1f}s"
        }
        
        mem_end = get_memory_mb()
        total_time = time.time() - req_start_time
        print(f"[{total_time:.3f}s] Memory Usage End: {mem_end:.2f} MB (Delta: {mem_end - mem_start:+.2f} MB)")
        print("=============================================\n")
        
        return str(output_path), summary

    except Exception as e:
        print("\n❌ CRITICAL ERROR IN PREDICT_VIDEO:")
        traceback.print_exc()
        print("=============================================\n")
        return None, {"Error": f"Backend failure: {str(e)}"}
