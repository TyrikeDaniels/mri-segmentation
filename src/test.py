import pickle
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import segmentation_models_pytorch as smp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test(model, loader, threshold=0.5):
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for data, target in loader:
            data   = data.to(device)
            target = target.to(device)
            output = torch.sigmoid(model(data))
            pred   = (output > threshold).float()
            intersection = (pred * target).sum(dim=(2, 3))
            union        = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
            batch_dice = (2 * intersection + 1e-6) / (union + 1e-6)
            dice_scores.extend(batch_dice.mean(dim=1).cpu().numpy())
    mean_dice = sum(dice_scores) / len(dice_scores)
    return mean_dice

def main():

    with open("data/loader_data.pkl", "rb") as f:
        loaders = pickle.load(f)

    model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=1, classes=1)
    model_path = Path(__file__).parent.parent / 'models' / 'best_model.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)

    train_dice = test(model, loaders["train"])
    test_dice  = test(model, loaders["test"])

    print(f"Train Dice: {train_dice:.4f}")
    print(f"Test Dice:  {test_dice:.4f}")
    print(f"Gap:        {train_dice - test_dice:.4f}")

if __name__ == "__main__":
    main()

