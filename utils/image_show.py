import matplotlib.pyplot as plt
import matplotlib.image as img

import numpy as np

def show_mri(data, indx, pred, num=3, alpha=0.5):
    fig, axs = plt.subplots(num, 3, figsize=(12, 4*num))
    
    for i in range(num):
        mri, ground_truth = data[indx[i]]

        mri = mri.squeeze(0)
        ground_truth = ground_truth.squeeze(0)

        mask = pred[indx[i]].squeeze(0)
        
        # MRI + mask overlay (Ground truth)
        axs[i, 0].imshow(mri, cmap='gray')
        axs[i, 0].imshow(ground_truth, alpha=alpha, cmap='hot')
        axs[i, 0].set_title(f"MRI + Mask (Ground truth)")
        
        # Mask + mask overlay (Unet)
        axs[i, 1].imshow(mri, cmap='gray')
        axs[i, 1].imshow(mask, alpha=alpha, cmap='hot')
        axs[i, 1].set_title(f"MRI + Mask (Prediction)")
        
        # MRI only
        axs[i, 2].imshow(mri, cmap='gray')
        axs[i, 2].set_title("MRI")
    
    plt.tight_layout()
    return fig