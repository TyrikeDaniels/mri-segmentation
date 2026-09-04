import sys
import os
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split 
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mri_dataset import BrainMRIDataset
from utils.transform import *

BATCH_SIZE = 8
RANDOM_STATE = 53


def main(): 
    image_dir = Path(__file__).parent.parent / 'data' / 'images'
    all_images = sorted(image_dir.glob('*.png'), key=lambda x: int(x.stem))
    
    full_images, test_images = train_test_split(all_images, test_size=0.2, random_state=RANDOM_STATE)
    train_images, val_images = train_test_split(full_images, test_size=0.2, random_state=RANDOM_STATE)

    train_data = BrainMRIDataset(train_images, train_transforms, mask_train_transforms)
    val_data = BrainMRIDataset(val_images, val_transforms, mask_val_transforms)
    test_data = BrainMRIDataset(test_images, val_transforms, mask_val_transforms)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    loaders = {
        "train": train_loader, 
        "val"  : val_loader, 
        "test" : test_loader
    }

    with open("data/loader_data.pkl", "wb") as f:
        pickle.dump(loaders, f)

if __name__ == "__main__":
    main()