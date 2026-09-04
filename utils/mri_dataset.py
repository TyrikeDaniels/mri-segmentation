import torch
from torch.utils.data import Dataset
from PIL import Image


class BrainMRIDataset(Dataset):

    def __init__(self, image_paths, train_trans=None, mask_trans=None):
        self.image_paths = image_paths
        self.train_trans = train_trans
        self.mask_trans = mask_trans

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        mask_path = image_path.parent.parent / 'masks' / image_path.name

        image = Image.open(image_path).convert("L")
        mask  = Image.open(mask_path).convert("L")

        seed = torch.randint(0, 2**32, (1,)).item()

        if self.train_trans:
            torch.manual_seed(seed)
            image = self.train_trans(image)

        if self.mask_trans:
            torch.manual_seed(seed)
            mask = self.mask_trans(mask)

        return image, mask