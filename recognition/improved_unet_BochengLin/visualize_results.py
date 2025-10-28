import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
import argparse
import os
import json

from dataset import Prostate3DDataset
from modules import UNet3D_Improved


def visualize_segmentation(model, test_loader, device, num_samples=3, save_dir=None):
    """
    Generate and save segmentation visualization images.
    Shows side-by-side comparison of input, ground truth, and predictions.
    """
    
    if save_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, "results")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.eval()
    
    sample_count = 0
    with torch.no_grad():
        for batch in test_loader:
            if sample_count >= num_samples:
                break
            
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            batch_size = images.shape[0]
            for b in range(batch_size):
                if sample_count >= num_samples:
                    break
                
                # Extract middle slice for visualization
                image = images[b, 0].cpu().numpy()
                label = labels[b, 0].cpu().numpy()
                pred = preds[b].cpu().numpy()
                
                mid_z = image.shape[2] // 2
                
                # Create comparison figure
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                
                # Input image
                axes[0].imshow(image[:, :, mid_z], cmap='gray')
                axes[0].set_title('Input Image (Middle Slice)')
                axes[0].axis('off')
                
                # Ground truth labels
                im1 = axes[1].imshow(label[:, :, mid_z], cmap='tab10', vmin=0, vmax=9)
                axes[1].set_title('Ground Truth Label')
                axes[1].axis('off')
                
                # Model prediction
                im2 = axes[2].imshow(pred[:, :, mid_z], cmap='tab10', vmin=0, vmax=9)
                axes[2].set_title('Model Prediction')
                axes[2].axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(save_dir, f"segmentation_example_{sample_count + 1}.png")
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                print(f"Saved: {save_path}")
                plt.close()
                
                sample_count += 1


def compute_metrics(model, test_loader, device, num_classes=6):
    """Compute Dice coefficient for each class on test set."""
    model.eval()
    all_dice_scores = {c: [] for c in range(num_classes)}
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1, keepdim=True)
            
            # Calculate Dice for each class
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


def plot_metrics(all_dice_scores, save_dir=None):
    """Generate and save Dice score bar chart."""
    
    if save_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, "results")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    classes = list(all_dice_scores.keys())
    mean_dices = [np.mean(all_dice_scores[c]) for c in classes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes, mean_dices, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on each bar
    for i, (c, dice) in enumerate(zip(classes, mean_dices)):
        ax.text(i, dice + 0.02, f'{dice:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Dice Coefficient', fontsize=12)
    ax.set_title('Per-Class Dice Scores on Test Set', fontsize=14)
    ax.set_ylim([0, 1.0])
    ax.axhline(y=0.7, color='red', linestyle='--', label='Target (0.7)', linewidth=2)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "per_class_dice.png")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_training_curves(save_dir=None):
    """Generate and save training curves from training_history.json."""
    
    if save_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, "results")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load training history
    history_path = os.path.join(save_dir, "training_history.json")
    if not os.path.exists(history_path):
        print(f"Warning: {history_path} not found. Skipping training curves.")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Extract data
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    val_dice = [h['val_dice'] for h in history]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss curves
    ax1.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)
    
    # Plot 2: Validation Dice Score
    ax2.plot(epochs, val_dice, 'g-^', label='Val Dice Score', linewidth=2, markersize=6)
    ax2.axhline(y=0.85, color='orange', linestyle='--', linewidth=2, label='Early Stopping Target (0.85)')
    ax2.axhline(y=0.70, color='red', linestyle='--', linewidth=2, label='Minimum Target (0.70)')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Dice Score', fontsize=12)
    ax2.set_title('Validation Dice Score', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    ax2.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    # Print summary statistics
    print("\n📊 Training Summary:")
    print(f"  • Total Epochs: {len(epochs)}")
    print(f"  • Final Train Loss: {train_loss[-1]:.4f}")
    print(f"  • Final Val Loss: {val_loss[-1]:.4f}")
    print(f"  • Final Val Dice: {val_dice[-1]:.4f}")
    print(f"  • Best Val Dice: {max(val_dice):.4f} (Epoch {epochs[val_dice.index(max(val_dice))]})")
    print(f"  • Early Stopping Target (0.85) Reached: {'✅ Yes' if max(val_dice) >= 0.85 else '❌ No'}\n")


def main(args):
    """Generate segmentation visualizations and performance metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create results directory if needed
    if args.save_dir is None:
        args.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load trained model
    model = UNet3D_Improved(in_channels=1, num_classes=6)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    print(f"Loaded model from {args.model_path}")
    
    # Load test dataset
    test_dataset = Prostate3DDataset(root_dir=args.data_path, split="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Generating visualizations for {args.num_samples} test samples...")
    
    # Generate segmentation visualizations
    visualize_segmentation(model, test_loader, device, args.num_samples, args.save_dir)
    
    # Plot training curves from training history
    print("Generating training curves...")
    plot_training_curves(args.save_dir)
    
    # Compute and plot Dice metrics
    print("Computing per-class Dice scores...")
    all_dice_scores = compute_metrics(model, test_loader, device, num_classes=6)
    plot_metrics(all_dice_scores, args.save_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("Results Summary")
    print("="*50)
    mean_dice_per_class = []
    for c in range(6):
        mean_dice = np.mean(all_dice_scores[c])
        mean_dice_per_class.append(mean_dice)
        print(f"Class {c}: {mean_dice:.4f}")
    
    overall_dice = np.mean(mean_dice_per_class)
    print(f"\nOverall Dice: {overall_dice:.4f}")
    print("="*50)
    print(f"Visualizations saved to {args.save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate segmentation visualizations and metrics")
    parser.add_argument("--data_path", type=str, default=r"C:\data\HipMRI_3D", help="Path to dataset root")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model (auto-detect if not provided)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to visualize")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Auto-detect model path if not provided
    if args.model_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.model_path = os.path.join(script_dir, "results", "best_model.pth")
    
    main(args)
