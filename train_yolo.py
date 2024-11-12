import os
import torch
from ultralytics import YOLO
from utils import remove_duplicate_labels, get_class_distribution
from utils import initialize_yolo_model, validate_yolo_model
from utils import DATASET_PATH, class_names

# Set up paths and configurations
device = 'cuda' if torch.cuda.is_available() else 'cpu'
YOLOV11X_MODEL_PATH = '/content/drive/MyDrive/Pallets_detection/YOLO_ObjectDetection_Dataset/models/yolo11x.pt'
data_yaml_path = os.path.join(DATASET_PATH, 'data.yaml')
save_dir = 'runs/pallet_detection'
max_epochs = 50
PATIENCE = 5
MIN_DELTA = 0.0005

# Ensure directories exist
os.makedirs(save_dir, exist_ok=True)

# Initialize the model
model = initialize_yolo_model(YOLOV11X_MODEL_PATH)

# Training configuration
training_config = {
    "data": data_yaml_path,
    "imgsz": 640,
    "batch": 8,
    "workers": 4,
    "device": device,
    "name": 'pallet_detection',
    "pretrained": True,
    "amp": True,
    "save_dir": save_dir,
    "lr0": 0.01,
    "lrf": 0.1,
    "optimizer": 'SGD',
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "freeze": 0,
}

# Early stopping variables
best_map = 0.0
patience_counter = 0

# Training loop
for epoch in range(1, max_epochs + 1):
    print(f"\nEpoch {epoch}/{max_epochs}")
    model.train(epochs=1, **training_config)

    # Validate the model
    current_map = validate_yolo_model(model, data_yaml_path, save_dir)
    print(f"Validation mAP@0.50:0.95: {current_map:.4f}")

    if current_map > best_map + MIN_DELTA:
        best_map = current_map
        patience_counter = 0
        best_model_path = os.path.join(save_dir, 'best_model.pt')
        model.save(best_model_path)
        print(f"Best model saved at: {best_model_path}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {PATIENCE} epochs without improvement.")
            break
    torch.cuda.empty_cache()
