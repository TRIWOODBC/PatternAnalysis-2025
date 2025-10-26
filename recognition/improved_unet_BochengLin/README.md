## Improved 3D U-Net for Prostate MRI Segmentation

**Author**: Bocheng Lin 48275565
**Project**: COMP3710 Pattern Analysis Report - Project 7 (Hard Difficulty)

## 1. Problem Description
This project implements a 6-class 3D segmenntation on the prostate 3D MRI dataset using an Improved UNet3D model. The goal is to achieve Dice >= 0.70 on all foreground classes.

## 2. Algorithm Description

Uses the BraTS 2017 Challenge 3D U-Net architecture with the following key features:

- **ResidualContextBlock**: Residual learning with 2×Conv3d blocks, Instance Normalization, LeakyReLU, and Dropout(0.3) between convolutions
- **Strided Convolutions**: Learnable downsampling with stride-2 convolutions instead of max-pooling
- **Trilinear Interpolation**: Upsampling without transposed convolutions (avoids checkerboard artifacts)
- **Instance Normalization**: Better than batch norm for small batch sizes (4 in this case)
- **LeakyReLU(0.01)**: Improved gradient flow compared to ReLU
- **Combined Loss**: 50% Dice² + 50% Focal Loss for better handling of hard examples and class imbalance
- **6-class Output**: Background + 5 tissue classes

### Model Details

- **Architecture**: 4-level encoder-decoder with 3 downsampling levels + bottleneck
- **Encoder**: Conv blocks at levels 1,2,3 + bottleneck (1 → 32 → 64 → 128 → 256 channels)
- **Decoder**: Symmetric upsampling with skip connections from encoder (256 → 128 → 64 → 32 → 6 classes)
- **Total Parameters**: ~8.9M (lightweight for 16GB GPU with batch_size=4)
- **Training Speed**: ~7 minutes per epoch

### Training Setup

- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-5)
- **Loss Function**: 0.5×DiceSquaredLoss + 0.5×FocalLoss (α=0.25, γ=2.0)
- **Learning Rate Schedule**: ReduceLROnPlateau (factor=0.5, patience=5 epochs)
- **Batch Size**: 4
- **Data Split**: 80% train / 10% val / 10% test (deterministic seed=42)
- **Early Stopping**: Stop when validation Dice ≥ 0.85
- **Max Epochs**: 100
- **Logging**: CSV + JSON training history with per-epoch metrics

## 3. How it Works

### Working Principle

The BraTS 2017 U-Net uses an encoder-decoder architecture with residual learning:

1. **Encoder** (3 levels + bottleneck): 
   - ResidualContextBlock: 2×Conv3d(3×3×3) + InstanceNorm + LeakyReLU + Dropout(0.3) + residual skip
   - DownsampleBlock: Stride-2 convolution for learnable downsampling (no pooling)
   - Progressively increases channels: 32 → 64 → 128 → 256

2. **Bottleneck** (deepest level):
   - ResidualContextBlock at 256 channels captures abstract volumetric features

3. **Decoder** (3 levels):
   - UpsampleBlock: Trilinear interpolation (scale factor 2) + Conv3d
   - Skip connections: Concatenate upsampled features with encoder features
   - ResidualContextBlock: Process concatenated features
   - Progressively decreases channels: 256 → 128 → 64 → 32

4. **Output layer**: 1×1×1 convolution produces 6-channel prediction (one per class)

During inference, argmax converts logits to class labels.

**Key Design Decisions:**
- **Residual Learning**: Skip connections improve gradient flow and feature reuse
- **Dropout(0.3)**: Regularization between Conv blocks to prevent overfitting
- **InstanceNorm**: Better than BatchNorm for small batch sizes
- **Trilinear Upsampling**: Smooth interpolation without checkerboard artifacts from transposed convolutions
- **Strided Convolutions**: Learnable downsampling preserves more information than fixed pooling
- **Combined Loss (Dice² + Focal)**: Focuses on hard examples and class boundaries

### Architecture Diagram

```
Input (1×128×128×64)
    ↓
enc1: ResidualContextBlock (32 ch)
    ↓
down1: DownsampleBlock stride=2 (32→64 ch)
    ↓
enc2: ResidualContextBlock (64 ch)
    ↓
down2: DownsampleBlock stride=2 (64→128 ch)
    ↓
enc3: ResidualContextBlock (128 ch)
    ↓
down3: DownsampleBlock stride=2 (128→256 ch)
    ↓
bottleneck: ResidualContextBlock (256 ch)
    ↓
up3: UpsampleBlock trilinear (256→128 ch)
    ↓
dec3: ResidualContextBlock (128+128→128 ch with skip)
    ↓
up2: UpsampleBlock trilinear (128→64 ch)
    ↓
dec2: ResidualContextBlock (64+64→64 ch with skip)
    ↓
up1: UpsampleBlock trilinear (64→32 ch)
    ↓
dec1: ResidualContextBlock (32+32→32 ch with skip)
    ↓
out_conv: Conv3d(1×1×1) → 6 classes
    ↓
Output (6×128×128×64)
```

### Loss Function

**DiceSquaredLoss**: Squares the Dice coefficient for harder penalty on small errors
```
Dice² = (2 * TP + ε) / (TP + FP + FN + ε)²
```

**FocalLoss**: Focuses on hard examples with α=0.25, γ=2.0
```
FL = -α(1-p)^γ log(p)
```

**Combined Loss**: 0.5×Dice² + 0.5×FocalLoss

### Segmentation Results

![Per-Class Dice Scores](results/per_class_dice.png)

![Segmentation Example 1](results/segmentation_example_1.png)

![Segmentation Example 2](results/segmentation_example_2.png)

![Segmentation Example 3](results/segmentation_example_3.png)

## 4. Dataset and Preprocessing

### Data

3D MRI volumes with labels (6 classes: background + 5 tissue types)

### Processing

- **Center crop/pad to 128×128×64 voxels**: Adaptive cropping for oversized volumes, zero-padding for undersized volumes, centered to preserve region-of-interest
- **Safe Z-score normalization**: Per-volume $(x - \mu) / \sigma$ with small constant (ε=1e-8) to handle near-constant volumes
- **Robust label file matching**: Automatically finds corresponding label files using flexible naming patterns (_LFOV, _MR, etc.)
- **Labels**: 0-5 (6 classes: background + 5 tissue types)

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

### Quick Start (Replace `/your/data/path` with your actual dataset path)

**Step 0: Prepare Your Dataset**

Create the folder structure and place your data:
```
/your/data/path/
├── semantic_MRs\           # Input 3D MRI volumes (*.nii.gz)
└── semantic_labels_only\   # Segmentation labels (*.nii.gz)
```

**Step 1: Train the Model**
```bash
python train.py --data_path /your/data/path --batch_size 4 --lr 1e-4 --epochs 100
```

**Step 2: Evaluate on Test Set**
```bash
python predict.py --data_path /your/data/path --batch_size 4
```

**Step 3: Generate Visualizations**
```bash
python visualize_results.py --data_path /your/data/path --num_samples 3
```

### Dataset Format

Your dataset should have this structure:

```
/your/data/path/
├── semantic_MRs\           # Input 3D MRI volumes
│   ├── case_001.nii.gz
│   ├── case_002.nii.gz
│   └── ...
└── semantic_labels_only\   # Segmentation labels
    ├── case_001_SEMANTIC.nii.gz
    ├── case_002_SEMANTIC.nii.gz
    └── ...
```

The dataset loader will automatically:
- Match image and label files by flexible naming patterns
- Split data 80/10/10 into train/val/test (deterministic, seed=42)
- Crop/pad volumes to 128×128×64
- Normalize using Z-score with safe handling

### Details for Each Step

**Training:**
- Initializes BraTS 2017 U-Net with Channel Attention (~8.9M parameters)
- Combined loss: 0.5×DiceSquaredLoss + 0.5×FocalLoss
- Adam optimizer (lr=1e-4, weight_decay=1e-5)
- Early stopping when validation Dice ≥ 0.85
- Saves: `best_model.pth`, `training_log.csv`, `training_history.json`
- Time: ~7 min/epoch on 16GB GPU

**Evaluation:**
- Computes per-class Dice on test set
- Model auto-detected from `results/best_model.pth`

**Visualization:**
- Generates segmentation examples and metrics plots
- Saved to `results/` directory

### Optional: Test Model Architecture

```bash
python test_model.py
```

Validates model forward/backward pass and compares attention vs non-attention variants.

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

- **Deterministic split**: Fixed random seed (seed=42) in data loading for reproducible train/val/test split
- **Model checkpointing**: Best model saved to `results/best_model.pth` based on validation Dice
- **Training logs**: 
  - `training_log.csv`: Per-epoch metrics for easy analysis in Excel/pandas
  - `training_history.json`: Complete history with timestamps for detailed tracking
- **Hyperparameter configuration**: All parameters configurable via command-line arguments
- **No random augmentation**: Preprocessing is deterministic (no augmentation during training or prediction)

## 9. References

- Isensee et al., "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge", arXiv:1802.10508, 2018

## 10. Acknowledgments

This project was developed with assistance from GitHub Copilot.
