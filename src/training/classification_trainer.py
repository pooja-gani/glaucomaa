import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
from src.models.classification import GlaucomaClassifier
from src.data.dataset import GlaucomaDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.utils.training_utils import EarlyStopping

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_classification(
    data_dir: str,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-4,
    use_cdr: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train SwinV2 for Glaucoma Classification.
    """
    logger.info(f"Starting Classification Training on {device}")
    
    # Transforms (include augmentation)
    train_transform = A.Compose([
        A.Resize(256, 256), # SwinV2 Tiny typically uses 256x256
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # Dataset
    full_dataset = GlaucomaDataset(data_dir, transform=train_transform)
    
    if len(full_dataset) == 0:
        logger.error("No images found.")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    # Ignoring use_cdr for now as per new model implementation
    model = GlaucomaClassifier(use_features=use_cdr, pretrained=True).to(device)
    
    # Loss: CrossEntropy (since model outputs logits for 2 classes)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr) # Swin usually prefers AdamW
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # Early Stopping
    os.makedirs("checkpoints", exist_ok=True)
    early_stopping = EarlyStopping(patience=5, verbose=True, path="checkpoints/classification_best.pth")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device).long() # CrossEntropy needs (B) LongTensor
            
            optimizer.zero_grad()
            
            if use_cdr:
                # Placeholder if we ever enable CDR
                cdr = torch.zeros((images.size(0), 1)).to(device)
                outputs = model(images, cdr)
            else:
                outputs = model(images)
            
            # Outputs are (B, 2) logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        # Validation
        val_acc, val_loss = validate(model, val_loader, criterion, device, use_cdr)
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        # Early Stopping check
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            logger.info("Early stopping triggered prior to max epochs.")
            break

def validate(model, loader, criterion, device, use_cdr):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device).long()
            
            if use_cdr:
                 cdr = torch.zeros((images.size(0), 1)).to(device)
                 outputs = model(images, cdr)
            else:
                 outputs = model(images)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Argmax for prediction
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return correct / total if total > 0 else 0, running_loss / len(loader)

if __name__ == "__main__":
    train_classification("data/processed", epochs=30)
