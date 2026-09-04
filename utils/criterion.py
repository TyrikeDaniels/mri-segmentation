import torch


def bce_dice_loss(pred, target, smooth=1e-6):

    bce = torch.nn.BCEWithLogitsLoss()(pred, target)

    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1 - ((2 * intersection + smooth) / (union + smooth)).mean()

    return bce + dice