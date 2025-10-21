## Improved 3D U-Net for Prostate MRI Segmentation

**Author**: Bocheng Lin 48275565
**Project**: COMP3710 Pattern Analysis Report - Project 7 (Hard Difficulty)

## 1. Problem Description
This project implements a 6-class 3D segmenntation on the prostate 3D MRI dataset using an Improved UNet3D model. The goal is to achieve Dice >= 0.70 on all foreground classes.

## 2. Algorithm Description

Uses a 3D U-Net architecture with the following improvements:

- **Encoder-Decoder with Skip Connections**: Standard U-Net structure but with strided convolutions (Conv3d stride=2) instead of max-pooling for downsampling
- **Instance Norm**: Works better than batch norm when batch size is small (4 in this case)
- **LeakyReLU**: Better gradient flow compared to ReLU
- **Dice Loss**: Optimizes directly for segmentation IoU, handles class imbalance well
- **6-class Output**: Background + 5 tissue classes

### Model Details

Encoder: 5 levels (1 → 64 → 128 → 256 → 320 channels)
Decoder: 4 levels with symmetric upsampling and skip connections
Total: ~17.5M parameters

### Training Setup

- Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
- Loss: Dice Loss
- Learning rate: Reduce by 0.5× if validation Dice doesn't improve for 5 epochs
- Batch size: 4
- Data split: 80% train / 10% val / 10% test
- Early stopping: None (run full 100 epochs)

## 3. How it Works

### Working Principle

The improved 3D U-Net uses an encoder-decoder architecture with key improvements over standard U-Net:

1. **Encoder** progressively downsamples using **strided convolutions (stride=2)** instead of max-pooling (improvement: learnable downsampling preserves more information)
2. **InstanceNorm + LeakyReLU** at each block (improvement: better for small batch sizes and smoother gradient flow)
3. **Bottleneck** captures abstract features at the deepest level (320 channels)
4. **Decoder** progressively upsamples with **ConvTranspose3d** and skip connections (recovers fine details)
5. **Output layer** produces 6-channel prediction (one per class)

During inference, argmax converts output to class labels.

**Key Improvements:**
- Strided convolutions: Learnable downsampling vs fixed pooling
- Instance normalization: Better than batch norm for small batches
- LeakyReLU: Avoids dead neurons, improves gradient flow

### Architecture Diagram

```
Input (1×128×128×64)
    ↓
[Conv + InstanceNorm + LeakyReLU] → Strided Conv (stride 2)
    ↓ (64 ch)
[Conv + InstanceNorm + LeakyReLU] → Strided Conv (stride 2)
    ↓ (128 ch)
[Conv + InstanceNorm + LeakyReLU] → Strided Conv (stride 2)
    ↓ (256 ch)
[Conv + InstanceNorm + LeakyReLU] → Strided Conv (stride 2)
    ↓ (320 ch)
[Conv + InstanceNorm + LeakyReLU] [Bottleneck]
    ↓ (320 ch)
ConvTranspose3d + Skip + [Conv Block] 
    ↓ (256 ch)
ConvTranspose3d + Skip + [Conv Block]
    ↓ (128 ch)
ConvTranspose3d + Skip + [Conv Block]
    ↓ (64 ch)
ConvTranspose3d + Skip + [Conv Block]
    ↓ (32 ch)
Output Conv (1×1×1, 6 classes)
    ↓
Output (6×128×128×64)
```

### Segmentation Results

![Per-Class Dice Scores](results/per_class_dice.png)

![Segmentation Example 1](results/segmentation_example_1.png)

![Segmentation Example 2](results/segmentation_example_2.png)

![Segmentation Example 3](results/segmentation_example_3.png)

## 4. Dataset and Preprocessing

### Data

3D MRI volumes with labels (6 classes: background + 5 tissue types)

### Processing

- Center crop to 128×128×64 voxels
- Z-score normalization per volume: $(x - \mu) / \sigma$
- Labels: 0-5 (one-hot for training)

### Split

- Train: 80% (~170 samples)
- Val: 10% (~21 samples)  
- Test: 10% (~21 samples)

Deterministic split for reproducibility.

## 5. Project Structure

- `dataset.py` - Prostate3DDataset class for loading and preprocessing 3D MRI data
- `modules.py` - UNet3D_Improved model architecture
- `train.py` - Training script with Dice loss and validation
- `predict.py` - Evaluation script for computing per-class Dice
- `visualize_results.py` - Generate segmentation visualizations and metrics
- `test_model.py` - Model validation script
- `best_model.pth` - Trained model weights
- `environment.yml` - Conda environment configuration

## 6. Usage

### Setup Environment

```bash
conda env create -f environment.yml
conda activate unet3d
```

### Training

```bash
python train.py --data_path C:\data\HipMRI_3D --batch_size 4 --lr 1e-4 --epochs 100
```

### Evaluation on Test Set

```bash
python predict.py --data_path C:\data\HipMRI_3D --model_path best_model.pth
```

### Generate Visualizations

```bash
python visualize_results.py --data_path C:\data\HipMRI_3D --num_samples 3
```

## 7. Dependencies

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.19.0
- nibabel >= 3.2.0
- tqdm >= 4.50.0
- matplotlib >= 3.3.0
- Python >= 3.8

See `environment.yml` for exact versions.

## 8. Reproducibility

- Fixed random seed in data splitting for deterministic train/val/test split
- Model weights saved to `best_model.pth`
- All hyperparameters configurable via command-line arguments
- Preprocessing is deterministic (no random augmentation in prediction)

## 9. References

- U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
- Instance Normalization: Ulyanov et al., 2016

## 10. Acknowledgments

This project was developed with assistance from GitHub Copilot.
