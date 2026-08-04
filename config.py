from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "detect" / "train2" / "weights" / "best.pt"

# Data directories
ASSETS_DIR = BASE_DIR / "assets"
DEMO_IMAGES_DIR = ASSETS_DIR / "demo_images"
DEMO_VIDEOS_DIR = ASSETS_DIR / "demo_videos"
CACHE_DIR = ASSETS_DIR / "cache"

# Create directories if they don't exist
for d in [ASSETS_DIR, DEMO_IMAGES_DIR, DEMO_VIDEOS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Metrics Configuration
EVALUATION_METRICS = {
    "Precision": "69.5%",
    "Recall": "77.9%",
    "mAP@0.5": "78.6%"
}

# Project Information
PROJECT_INFO = {
    "Algorithm": "YOLOv8n",
    "Framework": "PyTorch",
    "Learning Type": "Supervised Learning",
    "Epochs": "10",
    "Image Size": "416 × 416",
    "Classes": "2 Detection Classes",
    "Dataset": "Traffic Footage"
}

# Classes
CLASS_NAMES = {
    0: "With Helmet",
    1: "Without Helmet"
}
