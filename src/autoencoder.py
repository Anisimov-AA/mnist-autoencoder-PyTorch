"""Train autoencoder and save results"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# SETUP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# DATA PREPARATION

transform = transforms.Compose([transforms.ToTensor()])

mnist_data = datasets.MNIST(
    root='../data', 
    train=True,
    download=True,
    transform=transform
)

data_loader = torch.utils.data.DataLoader(
    dataset=mnist_data,
    batch_size=64,
    shuffle=True
)

# Data check
dataiter = iter(data_loader)
images, labels = next(dataiter)
print(f"Batch shape: {images.shape}")
print(f"Value range: [{torch.min(images):.3f}, {torch.max(images):.3f}]\n")

# MODEL

class Autoencoder(nn.Module):
    """
    Autoencoder: Compresses images to 8 numbers, then reconstructs them
    Architecture: 784 -> 8 -> 784
    """
    
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128), nn.ReLU(),   # ÷6
            nn.Linear(128, 64), nn.ReLU(),      # ÷2
            nn.Linear(64, 16), nn.ReLU(),       # ÷4
            nn.Linear(16, 4)                    # ÷4
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 28*28), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

# TRAINING SETUP

model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# TRAINING

num_epochs = 10
outputs = []

print("TRAINING")

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    
    for img, _ in data_loader:
        img = img.reshape(-1, 28*28).to(device)
        recon = model(img)
        loss = criterion(recon, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(data_loader)
    print(f'Epoch: {epoch+1:2d}/{num_epochs}, Avg Loss: {avg_loss:.4f}')
    outputs.append((epoch, img.cpu(), recon.cpu().detach()))

print("Training complete!\n")

# SAVE CHECKPOINT

os.makedirs('../checkpoints', exist_ok=True)
checkpoint_path = '../checkpoints/autoencoder_checkpoint.pth'

torch.save({
    'model_state_dict': model.state_dict(),
    'num_epochs': num_epochs,
    'final_loss': avg_loss,
    'outputs': outputs
}, checkpoint_path)

print(f"Results saved to {checkpoint_path}")