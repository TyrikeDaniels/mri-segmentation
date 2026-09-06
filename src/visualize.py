import pickle
from pathlib import Path

import torch
import segmentation_models_pytorch as smp


device = "cuda" if torch.cuda.is_available() else "cpu"

def 

def main():
    with open("data/loader_data.pkl", "rb") as f:
        loaders = pickle.load(f)

    model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=1, classes=1)
    model_path = Path(__file__).parent.parent / 'models' / 'best_model.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)

