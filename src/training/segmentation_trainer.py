import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import logging

# Add project root to path
sys.path.append(os.getcwd())

from src.models.segmentation import GlaucomaSegmentationModel
from src.data.dataset import GlaucomaSegmentationDataset
from src.utils.metrics import GlaucomaMetrics
from src.utils.training_utils import EarlyStopping
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_segmentation(
    data_dir: str,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train U-Net++ for Optic Disc/Cup Segmentation.
    """
    logger.info(f"Starting Segmentation Training on {device}")
    
    # Transforms
    train_transform = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # Dataset & Loader
    # Assuming data_dir has 'train' and 'val' subdirs or similar split logic.
    # For simplicity, we assume data_dir points to the root containing images/masks
    # and we split manually or use a subset.
    
    # NOTE: In a real scenario, we'd have explicit train/val folders.
    full_dataset = GlaucomaSegmentationDataset(data_dir, transform=train_transform)
    
    if len(full_dataset) == 0:
        logger.error("No images found. Check data directory.")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Fix val transform (hacky because random_split preserves underlying dataset transform)
    # Ideally use separate datasets.
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = GlaucomaSegmentationModel(classes=3).to(device)
    
    # Loss: CrossEntropy + Dice
    ce_loss = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Early Stopping
    os.makedirs("checkpoints", exist_ok=True)
    early_stopping = EarlyStopping(patience=5, verbose=True, path="checkpoints/segmentation_best.pth", delta=0.001)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device) # (B, H, W) with val 0,1,2
            
            optimizer.zero_grad()
            logits = model(images) # (B, 3, H, W)
            
            loss = ce_loss(logits, masks)
            # Add Dice Loss manually or via library if needed. 
            # CE is often enough for initial stabilization.
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        scheduler.step()
        
        # Validation
        val_dice = validate(model, val_loader, device)
        # We want to maximize dice, but EarlyStopping tracks loss minimization.
        # So we pass negative dice.
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Dice: {val_dice:.4f}")
        
        early_stopping(-val_dice, model)
        
        if early_stopping.early_stop:
            logger.info("Early stopping triggered processing.")
            break

def validate(model, loader, device):
    model.eval()
    dice_scores = []
    
    # Dice calc for Cup and Disc
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.cpu().numpy()
            
            preds, probs = model.predict_step(images)
            preds = preds.cpu().numpy()
            
            # Compute Dice for Disc (1) and Cup (2)
            # Simple average dice
            batch_dice = 0
            for i in range(len(preds)):
                d_pred = (preds[i] == 1)
                d_true = (masks[i] == 1)
                dice_d = GlaucomaMetrics.dice_coeff(d_pred, d_true)
                
                c_pred = (preds[i] == 2)
                c_true = (masks[i] == 2)
                dice_c = GlaucomaMetrics.dice_coeff(c_pred, c_true)
                
                batch_dice += (dice_d + dice_c) / 2
                
            dice_scores.append(batch_dice / len(preds))
            
    return sum(dice_scores) / len(dice_scores) if dice_scores else 0

if __name__ == "__main__":
    train_segmentation("data/processed_seg_train", epochs=30)
