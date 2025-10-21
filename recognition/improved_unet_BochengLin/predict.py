import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
from tqdm import tqdm

from dataset import Prostate3DDataset
from modules import UNet3D_Improved


def evaluate(model, test_loader, device, num_classes=6):
    """Evaluate model and compute per-class Dice coefficients."""
    model.eval()
    all_dice_scores = {c: [] for c in range(num_classes)}
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating", ncols=100)
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1, keepdim=True)
            
            for c in range(num_classes):
                pred_c = (preds == c).float()
                target_c = (labels == c).float()
                intersection = torch.sum(pred_c * target_c)
                union = torch.sum(pred_c) + torch.sum(target_c)
                
                if union == 0:
                    dice = 1.0
                else:
                    dice = (2.0 * intersection + 1.0) / (union + 1.0)
                    dice = dice.item()
                
                all_dice_scores[c].append(dice)
    
    return all_dice_scores


def main(args):
    """Load model and evaluate on test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = UNet3D_Improved(in_channels=1, num_classes=6)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    print(f"Loaded model from {args.model_path}")
    
    # Load test set
    test_dataset = Prostate3DDataset(root_dir=args.data_path, split="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    all_dice_scores = evaluate(model, test_loader, device, num_classes=6)
    
    # Print results
    print("\n" + "=" * 50)
    print("Test Set Results")
    print("=" * 50)
    
    mean_dice_per_class = []
    for c in range(6):
        mean_dice = np.mean(all_dice_scores[c])
        mean_dice_per_class.append(mean_dice)
        print(f"Class {c}: {mean_dice:.4f}")
    
    overall_dice = np.mean(mean_dice_per_class)
    print(f"\nOverall Dice: {overall_dice:.4f}")
    
    if overall_dice >= 0.7:
        print("✓ SUCCESS: Overall Dice >= 0.7")
    else:
        print(f"✗ FAIL: Overall Dice {overall_dice:.4f} < 0.7")
    
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on test set")
    parser.add_argument("--data_path", type=str, default=r"C:\data\HipMRI_3D", help="Path to dataset")
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    
    args = parser.parse_args()
    main(args)
