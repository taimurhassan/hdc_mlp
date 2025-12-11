import os
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_TO_IDX = {
    "AMD": 0,     # AMD (Drusen + CNV)
    "DME": 1,
    "NORMAL": 2,
}

class OCTDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=224, augment=False):
        """
        root_dir:
          e.g. /data/Zhang, /data/Duke, etc.
        Expect structure:
          root_dir/split/CLASS_NAME/*.png or *.jpg
        """
        self.root_dir = root_dir
        self.split = split
        self.samples = []

        for cls_name, idx in CLASS_TO_IDX.items():
            pattern = os.path.join(root_dir, split, cls_name, "*")
            files = glob(pattern)
            for f in files:
                self.samples.append((f, idx))

        if not self.samples:
            raise RuntimeError(f"No images found in {root_dir} for split {split}.")

        t_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
        if augment and split == "train":
            t_list.insert(0, transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(t_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label
