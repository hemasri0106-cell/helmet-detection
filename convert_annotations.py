import os
import shutil
import random
import xml.etree.ElementTree as ET

# =========================
# CONFIGURATION
# =========================

DATASET_PATH = "dataset_original"  # Raw extracted Kaggle dataset
OUTPUT_PATH = "dataset"            # YOLO formatted dataset output

TRAIN_SPLIT = 0.8  # 80% train, 20% validation

# Exact class names from your XML
CLASSES = ["With Helmet", "Without Helmet"]

# =========================
# Create YOLO Folder Structure
# =========================

for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(OUTPUT_PATH, folder), exist_ok=True)

# =========================
# Get XML Files
# =========================

annotations_path = os.path.join(DATASET_PATH, "annotations")
images_path = os.path.join(DATASET_PATH, "images")

xml_files = [f for f in os.listdir(annotations_path) if f.endswith(".xml")]

random.shuffle(xml_files)

split_index = int(len(xml_files) * TRAIN_SPLIT)
train_files = xml_files[:split_index]
val_files = xml_files[split_index:]

print(f"Total XML files: {len(xml_files)}")
print(f"Training files: {len(train_files)}")
print(f"Validation files: {len(val_files)}")

# =========================
# Convert XML to YOLO format
# =========================

def convert(xml_file, subset):
    tree = ET.parse(os.path.join(annotations_path, xml_file))
    root = tree.getroot()

    image_name = root.find("filename").text
    image_path = os.path.join(images_path, image_name)

    # Skip if image does not exist
    if not os.path.exists(image_path):
        print(f"Image not found: {image_name}")
        return

    # Copy image to train/val folder
    shutil.copy(image_path, os.path.join(OUTPUT_PATH, f"images/{subset}", image_name))

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    label_file = os.path.join(
        OUTPUT_PATH,
        f"labels/{subset}",
        os.path.splitext(image_name)[0] + ".txt"
    )

    with open(label_file, "w") as f:
        for obj in root.findall("object"):
            class_name = obj.find("name").text

            if class_name not in CLASSES:
                continue

            class_id = CLASSES.index(class_name)

            xmin = float(obj.find("bndbox/xmin").text)
            ymin = float(obj.find("bndbox/ymin").text)
            xmax = float(obj.find("bndbox/xmax").text)
            ymax = float(obj.find("bndbox/ymax").text)

            # Convert to YOLO format (normalized)
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

# =========================
# Run Conversion
# =========================

for xml_file in train_files:
    convert(xml_file, "train")

for xml_file in val_files:
    convert(xml_file, "val")

print("Conversion complete! YOLO dataset ready.")
