import torch
import numpy as np
from torch.utils.data import Dataset
import os
import nibabel as nib

class Prostate3DDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, target_shape=(128, 128, 64)):
        """
        Load 3D MRI dataset with semantic segmentation labels.
        
        Args:
            root_dir: Path to dataset root directory
            split: 'train', 'val', or 'test'
            transform: Optional data augmentation transforms
            target_shape: Output volume size (H, W, D)
        """
        self.img_dir = os.path.join(root_dir, "semantic_MRs")
        self.lbl_dir = os.path.join(root_dir, "semantic_labels_only")
        self.transform = transform
        self.target_shape = target_shape

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

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        img_path = os.path.join(self.img_dir, file_name)
        
        # Map image filename to corresponding label filename
        lbl_file_name = file_name.replace("_LFOV", "_SEMANTIC")
        lbl_path = os.path.join(self.lbl_dir, lbl_file_name)

        img = nib.load(img_path).get_fdata().astype(np.float32)
        lbl = nib.load(lbl_path).get_fdata().astype(np.int64)
        
        # Center crop to target shape
        h, w, d = img.shape
        th, tw, td = self.target_shape
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        z1 = int(round((d - td) / 2.))
        img = img[y1:y1+th, x1:x1+tw, z1:z1+td]
        lbl = lbl[y1:y1+th, x1:x1+tw, z1:z1+td]

        # Z-score normalization
        img = (img - img.mean()) / (img.std() + 1e-8)
        
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