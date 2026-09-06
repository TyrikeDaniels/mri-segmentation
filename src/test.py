import pickle
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_show import show_mri

import torch
import segmentation_models_pytorch as smp

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(model, loader, threshold=0.5):

    scores = []
    preds = []

    model.eval()
    with torch.no_grad():
        for data, target in loader:

            data = data.to(device)
            target = target.to(device)

            output = torch.sigmoid(model(data))
            pred = (output > threshold).float()

            intersection = (pred * target).sum(dim=(2, 3))
            union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
            batch_dice = (2 * intersection + 1e-6) / (union + 1e-6)

            scores.extend(batch_dice.mean(dim=1).cpu().numpy())
            preds.extend(pred.cpu())

    preds = torch.stack(preds)

    return scores, preds

def main():

    with open("data/loader_data.pkl", "rb") as f:
        loaders = pickle.load(f)

    model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=1, classes=1)
    model_path = Path(__file__).parent.parent / 'models' / 'best_model.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)

    train_scores, train_mask = test(model, loaders["train"])
    test_scores, test_mask  = test(model, loaders["test"])

    train_dice = sum(train_scores) / len(train_scores)
    test_dice = sum(test_scores) / len(test_scores)
    print(f"Train Dice: {train_dice:.4f}")
    print(f"Test Dice:  {test_dice:.4f}")

    train_data = list(loaders["train"])
    train_idx = np.argsort(train_scores)[::-1] # (reversed) sorted indicies

    flat_train_data = []
    for batch_mri, batch_mask in train_data:
        for i in range(batch_mri.shape[0]):  # Iterate through batch
            flat_train_data.append((batch_mri[i], batch_mask[i]))

    fig = show_mri(flat_train_data, train_idx, train_mask)
    fig.savefig('/visuals/segmentation_results.png', dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    main()

