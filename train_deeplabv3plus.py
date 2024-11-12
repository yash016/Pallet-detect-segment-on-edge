# train_deeplabv3plus.py

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
from utils import PalletDataset, get_transforms, decode_segmap
from torch.utils.data import DataLoader
import numpy as np

# Define DeepLabV3+ model
def create_deeplabv3plus(outputchannels=3, encoder_name='resnet101', encoder_weights='imagenet'):
    model = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=outputchannels, activation=None)
    return model

# Training function
def train_model(model, criterion, dataloaders, optimizer, metrics, device, num_epochs=50, patience=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_mean_iou = 0.0
    scaler = torch.cuda.amp.GradScaler()  # Automatic Mixed Precision
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')

        for phase in ['Train', 'Val']:
            if phase == 'Train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_ious = []

            for inputs, masks in dataloaders[phase]:
                inputs, masks = inputs.to(device), masks.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'Train'):
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, masks)

                    if phase == 'Train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

            # Early stopping condition
        print(f"Best Validation Mean IoU: {best_mean_iou:.4f}")
        model.load_state_dict(best_model_wts)
        return model

# Instantiate and configure the model, optimizer, and scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_deeplabv3plus().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
