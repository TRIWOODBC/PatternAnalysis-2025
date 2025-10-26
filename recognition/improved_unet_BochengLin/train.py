import torch
import torch.nn as nn
import torch.nn.functional as F
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


class DiceSquaredLoss(nn.Module):
    """
    Dice² Loss for multi-class segmentation (CAN3D variant).
    
    Squares the Dice coefficient to penalize small errors more heavily.
    Better stability and focus on boundary/hard regions compared to standard Dice.
    """
    def __init__(self, smooth=1.0, num_classes=6):
        super(DiceSquaredLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W, D) logits from model
            target: (B, 1, H, W, D) integer class labels
        Returns:
            Scalar loss value (1 - mean Dice²)
        """
        target_one_hot = torch.zeros_like(pred)
        for c in range(self.num_classes):
            target_one_hot[:, c] = (target.squeeze(1) == c).float()
        
        pred = torch.softmax(pred, dim=1)
        intersection = torch.sum(pred * target_one_hot, dim=(2, 3, 4))
        union = torch.sum(pred, dim=(2, 3, 4)) + torch.sum(target_one_hot, dim=(2, 3, 4))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        # Square the dice coefficient for harder penalty
        return 1.0 - (dice ** 2).mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.
    
    Down-weights easy negatives, focuses on hard positives and misclassified samples.
    Useful for medical imaging with foreground/background imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W, D) logits from model
            target: (B, 1, H, W, D) integer class labels
        Returns:
            Scalar focal loss value
        """
        # Reshape for cross-entropy
        pred_flat = pred.permute(0, 2, 3, 4, 1).contiguous()
        pred_flat = pred_flat.view(-1, self.num_classes)
        target_flat = target.squeeze(1).contiguous().view(-1).long()
        
        # Cross-entropy
        ce = F.cross_entropy(pred_flat, target_flat, reduction='none')
        
        # Focal term: (1 - p_t) ^ gamma
        probs = torch.exp(-ce)
        focal_weight = (1 - probs) ** self.gamma
        
        # Focal loss with alpha weighting
        focal = self.alpha * focal_weight * ce
        
        return focal.mean()


class CombinedLoss(nn.Module):
    """
    Combined Dice² + Focal Loss for CAN3D training.
    
    Balances volume-level accuracy (Dice²) with boundary/hard-sample focus (Focal).
    Recommended weighting: 0.5 * Dice² + 0.5 * Focal for equal contribution.
    """
    def __init__(self, dice_weight=0.5, focal_weight=0.5, num_classes=6):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceSquaredLoss(num_classes=num_classes)
        self.focal_loss = FocalLoss(num_classes=num_classes)

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W, D) logits
            target: (B, 1, H, W, D) labels
        Returns:
            Combined loss: dice_weight * Dice² + focal_weight * Focal
        """
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal


class DiceCoefficient:
    """
    Compute mean Dice coefficient for validation.
    
    Argmax predictions to get class assignments, then compute per-class Dice
    and average across all classes. Used to track segmentation quality during training.
    """
    def __init__(self, smooth=1.0, num_classes=6):
        self.smooth = smooth
        self.num_classes = num_classes

    def compute(self, pred, target):
        """
        Args:
            pred: (B, C, H, W, D) model logits
            target: (B, 1, H, W, D) integer labels
        Returns:
            Mean Dice score across all classes
        """
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
    """
    Train model for one epoch.
    
    Iterates through training batches, computes loss, and updates weights.
    Returns average loss across all batches.
    """
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
    """
    Validate model on validation set.
    
    Evaluates loss and Dice coefficient without gradient computation.
    Returns average loss and Dice score for the entire validation set.
    """
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
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            dice = dice_metric.compute(outputs, labels)
            total_dice += dice
            
            avg_dice = total_dice / (batch_idx + 1)
            pbar.set_postfix({"dice": f"{avg_dice:.4f}"})
    
    return total_loss / len(val_loader), total_dice / len(val_loader)


def main(args):
    """
    Train CAN3D for prostate MRI segmentation.
    Uses combined Dice² + Focal loss; logs metrics to CSV and JSON.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize model with 6 classes for prostate regions
    model = UNet3D_Improved(in_channels=1, num_classes=6)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup loss and optimization (CAN3D: Dice² + Focal)
    criterion = CombinedLoss(dice_weight=0.5, focal_weight=0.5, num_classes=6)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Load datasets with deterministic 80/10/10 split
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
    parser = argparse.ArgumentParser(description="Train CAN3D (Context Aggregation Network 3D) for prostate segmentation")
    parser.add_argument("--data_path", type=str, default=r"C:\data\HipMRI_3D", help="Path to dataset root directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    
    args = parser.parse_args()
    main(args)
