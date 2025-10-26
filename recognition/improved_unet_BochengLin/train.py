import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import numpy as np
import csv
import json
from datetime import datetime

from dataset import Prostate3DDataset
from modules import UNet3D_Improved


class DiceLoss(nn.Module):
    """Dice Loss for multi-class segmentation."""
    def __init__(self, smooth=1.0, num_classes=6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, pred, target):
        """pred: (B, C, H, W, D), target: (B, 1, H, W, D)"""
        target_one_hot = torch.zeros_like(pred)
        for c in range(self.num_classes):
            target_one_hot[:, c] = (target.squeeze(1) == c).float()
        
        pred = torch.softmax(pred, dim=1)
        intersection = torch.sum(pred * target_one_hot, dim=(2, 3, 4))
        union = torch.sum(pred, dim=(2, 3, 4)) + torch.sum(target_one_hot, dim=(2, 3, 4))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceCoefficient:
    """Compute Dice coefficient for validation."""
    def __init__(self, smooth=1.0, num_classes=6):
        self.smooth = smooth
        self.num_classes = num_classes

    def compute(self, pred, target):
        """pred: (B, C, H, W, D), target: (B, 1, H, W, D)"""
        pred = torch.argmax(pred, dim=1, keepdim=True)
        dice_scores = []
        
        for c in range(self.num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()
            intersection = torch.sum(pred_c * target_c)
            union = torch.sum(pred_c) + torch.sum(target_c)
            if union == 0:
                dice_scores.append(1.0)
            else:
                dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
                dice_scores.append(dice.item())
        
        return np.mean(dice_scores)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc="Training", ncols=100)
    
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate on val set and return loss and Dice."""
    model.eval()
    total_loss = 0.0
    dice_metric = DiceCoefficient(num_classes=6)
    total_dice = 0.0
    pbar = tqdm(val_loader, desc="Validating", ncols=100)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(images)
            
            # Debug: print shapes and value ranges
            if batch_idx == 0:
                print(f"\nDebug - Outputs shape: {outputs.shape}, range: [{outputs.min():.4f}, {outputs.max():.4f}]")
                print(f"Debug - Labels shape: {labels.shape}, unique values: {torch.unique(labels)}")
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            dice = dice_metric.compute(outputs, labels)
            total_dice += dice
            
            avg_dice = total_dice / (batch_idx + 1)
            pbar.set_postfix({"dice": f"{avg_dice:.4f}"})
    
    return total_loss / len(val_loader), total_dice / len(val_loader)


def main(args):
    """
    Train 3D U-Net model for prostate MRI segmentation.
    
    Training workflow:
    1. Initialize model and move to device (CPU/GPU)
    2. Setup Dice loss and Adam optimizer
    3. Load train/val datasets
    4. Train for specified epochs with validation
    5. Save best model based on validation Dice
    6. Use learning rate scheduling to adjust learning rate
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model - 6 classes (0,1,2,3,4,5)
    model = UNet3D_Improved(in_channels=1, num_classes=6)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = DiceLoss(num_classes=6)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Data loaders
    train_dataset = Prostate3DDataset(root_dir=args.data_path, split="train")
    val_dataset = Prostate3DDataset(root_dir=args.data_path, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Prepare results/logging directory
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Training loop
    best_dice = 0.0
    best_model_path = os.path.join(results_dir, "best_model.pth")

    # Logging files
    log_csv_path = os.path.join(results_dir, 'training_log.csv')
    history_json_path = os.path.join(results_dir, 'training_history.json')

    # If CSV doesn't exist, write header
    if not os.path.exists(log_csv_path):
        with open(log_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'epoch', 'train_loss', 'val_loss', 'val_dice', 'lr', 'is_best'])

    # In-memory history for JSON
    history = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Dice: {val_dice:.4f}")

        scheduler.step(val_dice)

        # Current learning rate
        try:
            current_lr = optimizer.param_groups[0]['lr']
        except Exception:
            current_lr = None

        # Determine if this is a new best
        is_best = False
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            is_best = True
            print(f"Saved best model with Dice: {best_dice:.4f}")

        # Append CSV log
        timestamp = datetime.now().isoformat()
        with open(log_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp, epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_dice:.6f}", f"{current_lr}", int(is_best)])

        # Append to in-memory history
        history.append({
            'timestamp': timestamp,
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'val_dice': float(val_dice),
            'lr': float(current_lr) if current_lr is not None else None,
            'is_best': bool(is_best)
        })
    
    # Save history JSON
    try:
        with open(history_json_path, 'w', encoding='utf-8') as jf:
            json.dump(history, jf, indent=2)
        print(f"Saved training history to: {history_json_path}")
    except Exception as e:
        print(f"Warning: could not save history JSON: {e}")

    print(f"\nTraining completed. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D U-Net for prostate segmentation")
    parser.add_argument("--data_path", type=str, default=r"C:\data\HipMRI_3D", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--target_dice", type=float, default=0.7, help="Target Dice coefficient")
    
    args = parser.parse_args()
    main(args)
