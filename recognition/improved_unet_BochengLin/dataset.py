import nibabel as nib
import torch
import numpy as np
from torch.utils.data import Dataset
import os

class Prostate3DDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.img_dir = os.path.join(root_dir, "semantic_MRs")
        self.lbl_dir = os.path.join(root_dir, "semantic_labels_only")
        self.imgs = sorted(os.listdir(self.img_dir))
        self.lbls = sorted(os.listdir(self.lbl_dir))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        lbl_path = os.path.join(self.lbl_dir, self.lbls[idx])
        img = nib.load(img_path).get_fdata().astype(np.float32)
        lbl = nib.load(lbl_path).get_fdata().astype(np.int64)
        # 归一化
        img = (img - img.mean()) / (img.std() + 1e-8)
        # 转为 Tensor (C, D, H, W)
        img = torch.from_numpy(img).unsqueeze(0)
        lbl = torch.from_numpy(lbl)
        return {"image": img, "label": lbl}
