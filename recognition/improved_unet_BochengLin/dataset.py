import torch
import numpy as np
from torch.utils.data import Dataset
import os
import nibabel as nib
# You might need libraries like these for resizing/augmentation
# import SimpleITK as sitk 
# from scipy.ndimage import zoom

class Prostate3DDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, target_shape=(128, 128, 64)):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): One of 'train', 'val', or 'test' to select the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_shape (tuple): The desired output shape (H, W, D) for cropping/padding.
        """
        self.img_dir = os.path.join(root_dir, "semantic_MRs")
        self.lbl_dir = os.path.join(root_dir, "semantic_labels_only")
        self.transform = transform
        self.target_shape = target_shape

        all_files = sorted(os.listdir(self.img_dir))
        
        # --- 1. IMPLEMENT DATA SPLITTING ---
        # Create a deterministic split (e.g., 80% train, 10% val, 10% test)
        np.random.seed(42) # for reproducibility
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
        
        # Convert image filename to label filename: replace _LFOV with _SEMANTIC
        lbl_file_name = file_name.replace("_LFOV", "_SEMANTIC")
        lbl_path = os.path.join(self.lbl_dir, lbl_file_name)

        img = nib.load(img_path).get_fdata().astype(np.float32)
        lbl = nib.load(lbl_path).get_fdata().astype(np.int64)
        
        # --- 2. IMPLEMENT UNIFORM INPUT SIZE ---
        # Placeholder for resizing/cropping/padding logic
        # For example, you could crop or pad to self.target_shape
        # This is a very basic center crop example. A more robust solution is needed.
        h, w, d = img.shape
        th, tw, td = self.target_shape
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        z1 = int(round((d - td) / 2.))
        img = img[y1:y1+th, x1:x1+tw, z1:z1+td]
        lbl = lbl[y1:y1+th, x1:x1+tw, z1:z1+td]

        # Normalize
        img = (img - img.mean()) / (img.std() + 1e-8)
        
        # Convert to Tensor (C, H, W, D) - assuming your model expects this
        # PyTorch standard is (C, D, H, W). Adjust if needed.
        img = torch.from_numpy(img).unsqueeze(0) 
        # --- 4. ADD CHANNEL DIM TO LABEL ---
        lbl = torch.from_numpy(lbl).unsqueeze(0)

        sample = {"image": img, "label": lbl}

        # --- 3. APPLY DATA AUGMENTATION ---
        if self.transform:
            sample = self.transform(sample)
            
        return sample
if __name__ == "__main__":
    import argparse

    # 1. Create argument parser
    parser = argparse.ArgumentParser(description="Test the Prostate3DDataset loader.")
    
    # 2. Add command line argument --data_path with default path
    parser.add_argument("--data_path", type=str, 
                        default=r"C:\data\HipMRI_3D",
                        help="Path to the root directory of the HipMRI dataset.")
    
    # 3. Parse arguments
    args = parser.parse_args()

    # 4. Use the parsed path to load data
    print(f" Loading data from: {args.data_path}")
    train_dataset = Prostate3DDataset(root_dir=args.data_path, split="train")
    
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print("Sample image shape:", sample["image"].shape)
        print("Sample label shape:", sample["label"].shape)