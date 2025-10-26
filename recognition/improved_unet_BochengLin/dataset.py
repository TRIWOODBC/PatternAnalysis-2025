import torch
import numpy as np
from torch.utils.data import Dataset
import os
import nibabel as nib
import warnings

class Prostate3DDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, target_shape=(128, 128, 64)):
        """
        Load 3D MRI dataset with semantic segmentation labels.
        
        Args:
            root_dir: Path to dataset root directory
            split: 'train', 'val', or 'test'
            transform: Optional data augmentation transforms
            target_shape: Output volume size (H, W, D) for center crop/pad
        """
        self.img_dir = os.path.join(root_dir, "semantic_MRs")
        self.lbl_dir = os.path.join(root_dir, "semantic_labels_only")
        self.transform = transform
        self.target_shape = target_shape

        # Verify directories exist
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.isdir(self.lbl_dir):
            raise FileNotFoundError(f"Label directory not found: {self.lbl_dir}")

        all_files = sorted(os.listdir(self.img_dir))
        
        # Deterministic train/val/test split (80/10/10)
        np.random.seed(42)
        np.random.shuffle(all_files)
        
        train_split = int(0.8 * len(all_files))
        val_split = int(0.9 * len(all_files))

        if split == "train":
            self.file_list = all_files[:train_split]
        elif split == "val":
            self.file_list = all_files[train_split:val_split]
        elif split == "test":
            self.file_list = all_files[val_split:]
        else:
            raise ValueError(f"Invalid split name: {split}")

        print(f"Loaded {len(self.file_list)} files for the {split} set.")

    def _find_label_file(self, img_filename):
        """
        Robustly find corresponding label file for an image file.
        
        Tries multiple naming conventions:
        1. Replace _LFOV with _SEMANTIC
        2. Replace _MR with _SEMANTIC
        3. Exact basename match with different extension
        """
        # Try common replacement patterns
        candidates = [
            img_filename.replace("_LFOV", "_SEMANTIC"),
            img_filename.replace("_MR", "_SEMANTIC"),
        ]
        
        for candidate in candidates:
            if os.path.exists(os.path.join(self.lbl_dir, candidate)):
                return candidate
        
        # If no pattern matches, raise error with helpful message
        available_labels = os.listdir(self.lbl_dir)
        raise FileNotFoundError(
            f"Cannot find label for '{img_filename}'. "
            f"Tried: {candidates}. "
            f"Available labels (first 5): {available_labels[:5]}"
        )

    def _center_crop_or_pad(self, volume, target_shape):
        """
        Center crop or pad volume to target shape.
        
        Args:
            volume: Input 3D array (H, W, D)
            target_shape: Target shape (H, W, D)
            
        Returns:
            Resized volume (target_shape)
        """
        current_shape = volume.shape
        output = np.zeros(target_shape, dtype=volume.dtype)
        
        # Calculate slices for source (where to crop from input)
        src_slices = []
        dst_slices = []
        
        for i in range(3):
            if current_shape[i] >= target_shape[i]:
                # Crop: center the crop region
                start = (current_shape[i] - target_shape[i]) // 2
                src_slices.append(slice(start, start + target_shape[i]))
                dst_slices.append(slice(None))
            else:
                # Pad: center the input in the output
                start = (target_shape[i] - current_shape[i]) // 2
                src_slices.append(slice(None))
                dst_slices.append(slice(start, start + current_shape[i]))
        
        output[tuple(dst_slices)] = volume[tuple(src_slices)]
        return output

    def _normalize_safe(self, volume, eps=1e-8):
        """
        Safe z-score normalization that handles near-constant volumes.
        
        Args:
            volume: Input array
            eps: Small constant to avoid division by near-zero
            
        Returns:
            Normalized array
        """
        mean = volume.mean()
        std = volume.std()
        
        # Warn if std is too small (potentially constant volume)
        if std < eps:
            warnings.warn(f"Volume has very small std ({std:.2e}), clamping to {eps}")
            std = eps
        
        return (volume - mean) / (std + eps)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        img_path = os.path.join(self.img_dir, file_name)
        
        # Robustly find corresponding label file
        lbl_file_name = self._find_label_file(file_name)
        lbl_path = os.path.join(self.lbl_dir, lbl_file_name)

        img = nib.load(img_path).get_fdata().astype(np.float32)
        lbl = nib.load(lbl_path).get_fdata().astype(np.int64)
        
        # Center crop or pad to target shape (handles both cases)
        img = self._center_crop_or_pad(img, self.target_shape)
        lbl = self._center_crop_or_pad(lbl, self.target_shape)

        # Safe z-score normalization
        img = self._normalize_safe(img)
        
        # Convert to tensor with channel dimension (1, H, W, D)
        img = torch.from_numpy(img).unsqueeze(0)
        lbl = torch.from_numpy(lbl).unsqueeze(0)

        sample = {"image": img, "label": lbl}

        # Optional data augmentation
        if self.transform:
            sample = self.transform(sample)
            
        return sample
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("--data_path", type=str, 
                        default=r"C:\data\HipMRI_3D",
                        help="Path to dataset root directory")
    
    args = parser.parse_args()

    print(f"Loading data from: {args.data_path}")
    train_dataset = Prostate3DDataset(root_dir=args.data_path, split="train")
    
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print("Sample image shape:", sample["image"].shape)
        print("Sample label shape:", sample["label"].shape)