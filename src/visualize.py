"""Visualize saved training results"""

import os
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# SETUP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = '../checkpoints/autoencoder_checkpoint.pth'

# LOAD CHECKPOINT

if not os.path.exists(checkpoint_path):
    print(f"Checkpoint not found: {checkpoint_path}")
    print("Run src/autoencoder.py first to train the model!")
    exit()

print(f"Loading results from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

num_epochs = checkpoint['num_epochs']
final_loss = checkpoint['final_loss']
outputs = checkpoint['outputs']

print(f"Loaded results from {num_epochs} epochs")
print(f"Final loss: {final_loss:.4f}\n")

# VISUALIZATION

epochs_to_show = [0, num_epochs//2, num_epochs-1]

fig = plt.figure(figsize=(18, 9))
gs_main = gridspec.GridSpec(len(epochs_to_show), 1, figure=fig, hspace=0.4)

for epoch_idx, k in enumerate(epochs_to_show):
    epoch, imgs, recon = outputs[k]
    
    # Convert to CPU first, then numpy
    imgs = imgs.cpu().numpy()
    recon = recon.cpu().numpy()
    
    gs_epoch = gridspec.GridSpecFromSubplotSpec(
        2, 9, subplot_spec=gs_main[epoch_idx], hspace=0.05, wspace=0.05
    )
    
    # Original images
    for i in range(9):
        ax = fig.add_subplot(gs_epoch[0, i])
        ax.imshow(imgs[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_ylabel(f'Epoch {epoch+1}\nOriginal', rotation=0,  # ← Исправлено!
                         fontsize=11, labelpad=50, va='center', fontweight='bold')
    
    # Reconstructed images
    for i in range(9):
        ax = fig.add_subplot(gs_epoch[1, i])
        ax.imshow(recon[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('Reconstructed', rotation=0, 
                         fontsize=11, labelpad=50, va='center')

plt.suptitle(f'Training Progress: Epochs 1, {num_epochs//2 + 1}, and {num_epochs}',
             fontsize=15, y=0.98, fontweight='bold')
plt.show()