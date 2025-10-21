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

### Project Structure

- `dataset.py` - Prostate3DDataset class for loading and preprocessing 3D MRI data
- `modules.py` - UNet3D_Improved model architecture with ConvBlock, DownBlock, UpBlock components
- `train.py` - Training script with Dice loss, validation, and model checkpointing
- `predict.py` - Inference script for evaluating on test set
- `test_model.py` - Model validation script

### Environment Setup

Create the conda environment:

```

### Model Architecture

The UNet3D_Improved model consists of:

- **Encoder**: 5 levels with channel progression 32→64→128→256→320
  - Uses strided convolutions for downsampling instead of max-pooling
  - Each level has double convolution blocks with InstanceNorm and LeakyReLU
  
- **Decoder**: 4 levels with symmetric upsampling
  - Uses ConvTranspose3d for upsampling
  - Skip connections concatenate encoder features with decoder features
  - Automatically handles odd dimension padding

- **Output**: 1x1x1 convolution for final segmentation map

Total parameters: ~17.5M

### Dataset

The dataset is split into train/val/test sets (80/10/10 ratio):

- Images: 128×128×64 (3D volumes, center-cropped)
- Labels: Semantic segmentation masks
- Preprocessing: Normalization (z-score) applied to images

Data loading:

```bash
python dataset.py --data_path /path/to/HipMRI_3D
```

### Training

Train the model on train/val sets:

```bash
python train.py --data_path /path/to/HipMRI_3D --batch_size 4 --lr 1e-4 --epochs 100
```

Options:
- `--data_path`: Path to dataset root directory
- `--batch_size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 1e-4)
- `--epochs`: Number of epochs (default: 100)
- `--target_dice`: Target Dice coefficient for early stopping (default: 0.7)

The training script:
- Uses Dice loss for multi-class segmentation
- Saves best model to `best_model.pth`
- Logs training/validation metrics with progress bars
- Uses learning rate scheduling (ReduceLROnPlateau)

### Evaluation

Evaluate the trained model on test set:

```bash
python predict.py --data_path /path/to/HipMRI_3D --model_path best_model.pth
```

The evaluation script computes per-class Dice coefficients and overall performance.bash
conda env create -f environment.yml
conda activate unet3d
