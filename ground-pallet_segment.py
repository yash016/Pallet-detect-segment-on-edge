# ground-pallet_segment.py

import os
import torch
from utils import PalletDataset, get_transforms, decode_segmap
from train_deeplabv3plus import train_model, create_deeplabv3plus
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# Set dataset paths
HOME_DIR = "/content/drive/MyDrive/Pallets_detection"
SEG_DATASET_DIR = os.path.join(HOME_DIR, "Deeplabv3_ObjectSegmentation_Dataset")
train_images_dir = os.path.join(SEG_DATASET_DIR, "train", "images")
train_masks_dir = os.path.join(SEG_DATASET_DIR, "train", "masks")
val_images_dir = os.path.join(SEG_DATASET_DIR, "val", "images")
val_masks_dir = os.path.join(SEG_DATASET_DIR, "val", "masks")

# Set up transformations and dataloaders
input_size = 512
train_image_transform, train_mask_transform, val_image_transform, val_mask_transform = get_transforms(input_size)

train_dataset = PalletDataset(train_images_dir, train_masks_dir, train_image_transform, train_mask_transform)
val_dataset = PalletDataset(val_images_dir, val_masks_dir, val_image_transform, val_mask_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

dataloaders = {'Train': train_loader, 'Val': val_loader}

# Train model
model = create_deeplabv3plus().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
trained_model = train_model(model, dataloaders)

# Save the model
model_dir = os.path.join(SEG_DATASET_DIR, "model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "best_deeplabv3plus_model.pth")
torch.save(trained_model.state_dict(), model_path)

# Visualize test results
# Code for loading the test set, predicting, and visualizing results would follow here.
