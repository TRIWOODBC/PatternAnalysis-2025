## Improved 3D U-Net for Prostate MRI Segmentation

This project implements a 3D improved U-Net architecture for semantic segmentation on 3D medical imaging data.

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
