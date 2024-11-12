# utils.py

import os
import cv2
import torch
import numpy as np
import random
import glob
import shutil
import xml.etree.ElementTree as ET
import logging
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import (Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip,
                                    RandomVerticalFlip, RandomRotation, ColorJitter, RandomResizedCrop, InterpolationMode)
import albumentations as A
from ultralytics import YOLO
from IPython.display import display
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import Model
import torchvision.ops as ops

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CLASSES = ['floor', 'ground', 'pavement', 'surface', 'carpet', 'tile', 'pallet']
ground_aliases = {'carpet', 'floor', 'pavement', 'surface', 'tile'}
class_mapping = {'ground': 0, 'pallet': 1}

# ========== Data Augmentation and Transformations ==========

augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.RandomBrightnessContrast(p=0.7),
    A.HueSaturationValue(p=0.5),
    A.RGBShift(p=0.3),
    A.RandomGamma(p=0.3),
    A.Blur(blur_limit=3, p=0.3),
    A.GaussNoise(p=0.2),
    A.ElasticTransform(p=0.2),
    A.GridDistortion(p=0.2)
])

def augment_and_save_images(original_dir, augmented_dir, num_augmentations=5):
    """
    Apply augmentations to images and save them in the augmented directory.
    """
    image_filenames = [
        f for f in os.listdir(original_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and "_aug" not in f.lower()
    ]
    for image_filename in tqdm(image_filenames, desc="Augmenting Images"):
        image_path = os.path.join(original_dir, image_filename)
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Failed to read image: {image_path}. Skipping.")
            continue
        for aug_idx in range(1, num_augmentations + 1):
            augmented = augmentation_pipeline(image=image)
            augmented_image = augmented['image']
            base_name, ext = os.path.splitext(image_filename)
            augmented_image_filename = f"{base_name}_aug{aug_idx}{ext}"
            augmented_image_path = os.path.join(augmented_dir, augmented_image_filename)
            cv2.imwrite(augmented_image_path, augmented_image)
    logger.info(f"All augmented images saved in: {augmented_dir}")

def get_transforms(input_size):
    train_image_transform = Compose([
        RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(10),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = Compose([
        RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=InterpolationMode.NEAREST),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(10, interpolation=InterpolationMode.NEAREST),
    ])

    val_image_transform = Compose([
        Resize((input_size, input_size), interpolation=InterpolationMode.BILINEAR),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_mask_transform = Resize((input_size, input_size), interpolation=InterpolationMode.NEAREST)
    return train_image_transform, mask_transform, val_image_transform, val_mask_transform

# ========== Custom Dataset for Segmentation ==========

class PalletDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = map_mask(mask)
        mask = torch.from_numpy(mask).long()
        return image, mask

# Map pixel values in masks to class indices
def map_mask(mask):
    mask = np.array(mask)
    mask = np.where(mask == 128, 1, mask)  # Ground -> 1
    mask = np.where(mask == 255, 2, mask)  # Pallet -> 2
    mask = np.where((mask != 1) & (mask != 2), 0, mask)  # Background -> 0
    return mask

# Post-process segmentation mask
def post_process(mask):
    mask = mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# Decode segmented mask for visualization
def decode_segmap(image, num_classes):
    label_colors = np.array([(0, 0, 0), (128, 128, 128), (255, 255, 255)])
    r, g, b = [np.zeros_like(image).astype(np.uint8) for _ in range(3)]
    for l in range(num_classes):
        idx = image == l
        r[idx], g[idx], b[idx] = label_colors[l]
    return np.stack([r, g, b], axis=2)

# ========== Model Utilities ==========

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    if xyxy.ndim == 1 and xyxy.size == 4:
        xyxy = xyxy.reshape(1, 4)
    elif xyxy.ndim == 1 and xyxy.size == 0:
        xyxy = np.empty((0, 4))
    elif xyxy.ndim != 2:
        raise ValueError(f"Unexpected shape for xyxy: {xyxy.shape}")

    for box in xyxy:
        try:
            masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
            if masks.size == 0:
                continue
            result_masks.append(masks[np.argmax(scores)])
        except Exception as e:
            logger.error(f"Error processing box {box}: {e}")
    return np.array(result_masks) if result_masks else np.empty((0, *image.shape[:2]), dtype=np.uint8)

def apply_nms(detections, iou_threshold=0.9):
    if len(detections.xyxy) == 0:
        return detections
    boxes = torch.tensor(detections.xyxy, dtype=torch.float32)
    scores = torch.tensor(detections.confidence, dtype=torch.float32)
    keep_indices = ops.nms(boxes, scores, iou_threshold)
    keep_indices = keep_indices.cpu().numpy()
    detections.xyxy = detections.xyxy[keep_indices]
    detections.confidence = detections.confidence[keep_indices]
    detections.class_id = detections.class_id[keep_indices]
    if hasattr(detections, 'mask') and detections.mask is not None:
        detections.mask = detections.mask[keep_indices]
    return detections

def initialize_models(home_dir):
    dino_config = os.path.join(home_dir, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    dino_checkpoint = os.path.join(home_dir, "weights", "groundingdino_swint_ogc.pth")
    sam_checkpoint = os.path.join(home_dir, "weights", "sam_vit_h_4b8939.pth")
    grounding_dino_model = Model(model_config_path=dino_config, model_checkpoint_path=dino_checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device=device)
    sam_predictor = SamPredictor(sam)
    return grounding_dino_model, sam_predictor

def convert_xml_to_yolo(merged_xml_path, yolo_txt_path, class_mapping):
    tree = ET.parse(merged_xml_path)
    root = tree.getroot()
    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)
    yolo_annotations = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text.lower().strip()
        if class_name in ground_aliases:
            class_name = 'ground'
        class_id = class_mapping.get(class_name)
        if class_id is None:
            print(f"Warning: Unrecognized class '{class_name}' in file {merged_xml_path}. Skipping.")
            continue
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        x_center = round((xmin + xmax) / 2 / img_width, 4)
        y_center = round((ymin + ymax) / 2 / img_height, 4)
        width = round((xmax - xmin) / img_width, 4)
        height = round((ymax - ymin) / img_height, 4)
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
    with open(yolo_txt_path, 'w') as f:
        f.write('\n'.join(yolo_annotations))

def create_mask_from_xml(merged_xml_path, mask_path, class_mapping):
    tree = ET.parse(merged_xml_path)
    root = tree.getroot()
    size = root.find('size')
    img_width = int(size.find('width').text) if size is not None else 416
    img_height = int(size.find('height').text) if size is not None else 416
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for obj in root.findall('object'):
        class_name = obj.find('name').text.strip().lower()
        if class_name in ground_aliases:
            class_name = 'ground'
        class_id = class_mapping.get(class_name)
        if class_id is None:
            print(f"Class '{class_name}' not in class_mapping. Skipping.")
            continue
        polygon = obj.find('polygon')
        points = []
        if polygon is not None:
            for pt in polygon:
                x = int(pt.find('x').text)
                y = int(pt.find('y').text)
                points.append((x, y))
            if points:
                cv2.fillPoly(mask, [np.array(points, np.int32)], class_id)
    cv2.imwrite(mask_path, mask)

def split_dataset(images_dir, labels_dir, output_dir, split_ratios=(0.8, 0.1, 0.1)):
    split_dirs = {
        "train": {"images": os.path.join(output_dir, "train", "images"), "labels": os.path.join(output_dir, "train", "labels")},
        "val": {"images": os.path.join(output_dir, "val", "images"), "labels": os.path.join(output_dir, "val", "labels")},
        "test": {"images": os.path.join(output_dir, "test", "images"), "labels": os.path.join(output_dir, "test", "labels")},
    }
    for split in split_dirs.values():
        os.makedirs(split["images"], exist_ok=True)
        os.makedirs(split["labels"], exist_ok=True)
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files)
    num_images = len(image_files)
    train_size = int(split_ratios[0] * num_images)
    val_size = int(split_ratios[1] * num_images)
    splits = {
        "train": image_files[:train_size],
        "val": image_files[train_size:train_size + val_size],
        "test": image_files[train_size + val_size:]
    }
    for split, files in splits.items():
        for image_file in files:
            image_src = os.path.join(images_dir, image_file)
            label_file = image_file.rsplit(".", 1)[0] + ".txt"
            label_src = os.path.join(labels_dir, label_file)
            image_dst = os.path.join(split_dirs[split]["images"], image_file)
            label_dst = os.path.join(split_dirs[split]["labels"], label_file)
            shutil.copy(image_src, image_dst)
            if os.path.exists(label_src):
                shutil.copy(label_src, label_dst)
            else:
                logger.warning(f"Annotation file {label_src} does not exist for image {image_file}")
    logger.info("Dataset split completed successfully!")
