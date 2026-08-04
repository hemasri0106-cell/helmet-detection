import shutil
import random
from pathlib import Path
import config
from inference import predict_image
import cv2

def setup_videos():
    print("Setting up demo videos...")
    video_map = {
        "traffic_video.mp4": "Busy Market.mp4",
        "in_traffic.mp4": "City Road Ride.mp4",
        "video_with_helmets.mp4": "Highway Ride.mp4"
    }
    
    for src_name, dst_name in video_map.items():
        src_path = config.BASE_DIR / src_name
        dst_path = config.DEMO_VIDEOS_DIR / dst_name
        
        if src_path.exists() and not dst_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"Copied {src_name} to {dst_name}")

def setup_images():
    print("Setting up demo images...")
    base_images = ["image10.jpg", "bus.jpg", "test.jpg"]
    
    # Try to copy known images
    copied_count = 0
    for img_name in base_images:
        src_path = config.BASE_DIR / img_name
        if src_path.exists():
            dst_path = config.DEMO_IMAGES_DIR / img_name
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
                print(f"Copied {img_name}")
            copied_count += 1
            
    # If we don't have enough (at least 4), try to find some in dataset
    if copied_count < 4:
        dataset_images_dir = config.BASE_DIR / "dataset" / "images" / "val"
        if dataset_images_dir.exists():
            images = list(dataset_images_dir.glob("*.jpg"))
            if images:
                random.shuffle(images)
                for img_path in images[:4 - copied_count]:
                    dst_path = config.DEMO_IMAGES_DIR / img_path.name
                    if not dst_path.exists():
                        shutil.copy2(img_path, dst_path)
                        print(f"Copied {img_path.name} from dataset")

def cache_dashboard_inferences():
    print("Caching dashboard inferences...")
    # Pre-run inference on demo images so the dashboard can load quickly
    for img_path in config.DEMO_IMAGES_DIR.glob("*.jpg"):
        cache_out = config.CACHE_DIR / f"dash_{img_path.name}"
        print(f"Running inference for {img_path.name}...")
        annotated_img, _ = predict_image(img_path)
        if annotated_img is not None:
            # Convert RGB back to BGR for saving with cv2
            bgr_img = annotated_img[:, :, ::-1] if annotated_img.shape[-1] == 3 else annotated_img
            cv2.imwrite(str(cache_out), bgr_img)
            print(f"Cached {cache_out.name}")

if __name__ == "__main__":
    setup_videos()
    setup_images()
    cache_dashboard_inferences()
    print("Setup complete!")
