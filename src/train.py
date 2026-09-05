from pathlib import Path
import sys
import os
import pickle

import torch
from tqdm import tqdm
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.criterion import bce_dice_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, criterion, optimizer, scheduler, epochs, loaders):

    model_path = Path(__file__).parent.parent / 'models' / 'best_model.pth'
    model_path.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = loaders
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        train_loss = train_loss / len(train_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print("save new model")
        print(f'[{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

def main():

    model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=1, classes=1)
    model.to(device)

    with open("data/loader_data.pkl", "rb") as f:
        loaders = pickle.load(f)

    l_r = 0.01
    epochs = 50
    w_d = 1e-4
    m = 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=l_r, momentum=m, weight_decay=w_d)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=l_r,
        steps_per_epoch=len(loaders["train"]),
        epochs=epochs,
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100
    )

    train(
        model=model, 
        criterion=bce_dice_loss, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        epochs=epochs, 
        loaders=(loaders["train"], loaders["val"])
    )

if __name__ == "__main__":
    main()