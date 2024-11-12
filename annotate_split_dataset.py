# annotate.py

import os
import cv2
from tqdm import tqdm
from utils import (
    initialize_models, 
    augment_and_save_images, 
    segment, 
    apply_nms, 
    split_dataset, 
    logger
)

# Set up directories
HOME = "/content/drive/MyDrive/Pallets_detection"
PALLETS_DIR = os.path.join(HOME, "Pallets")
ANNOTATED_IMAGES_DIR = os.path.join(HOME, "Annotations", "Annotated_Images")
YOLO_LABELS_DIR = os.path.join(HOME, "YOLO_Annotations")
SPLIT_OUTPUT_DIR = os.path.join(HOME, "YOLO_ObjectDetection_Dataset")

# Initialize models
grounding_dino_model, sam_predictor = initialize_models(HOME)

# Process images
image_paths = [
    os.path.join(PALLETS_DIR, f) for f in os.listdir(PALLETS_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]
for image_path in tqdm(image_paths, desc="Processing images"):
    image_name = os.path.basename(image_path)
    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"Failed to read image: {image_path}")
        continue

    # Perform augmentation (optional)
    augment_and_save_images(PALLETS_DIR, PALLETS_DIR, num_augmentations=5)

    # Perform detection and segmentation
    detections = grounding_dino_model.predict_with_classes(image=image, classes=CLASSES)
    detections = apply_nms(detections)
    segmented_masks = segment(sam_predictor=sam_predictor, image=image, xyxy=detections.xyxy)

    # Annotate and save the result
    annotated_save_path = os.path.join(ANNOTATED_IMAGES_DIR, f"annotated_{image_name}")
    cv2.imwrite(annotated_save_path, image)
    logger.info(f"Annotated image saved at {annotated_save_path}")

# Split the dataset into train, val, and test sets
split_dataset(
    images_dir=PALLETS_DIR, 
    labels_dir=YOLO_LABELS_DIR, 
    output_dir=SPLIT_OUTPUT_DIR, 
    split_ratios=(0.8, 0.1, 0.1)
)

logger.info("Dataset split completed successfully.")
