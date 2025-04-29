import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import timm

class VADDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['path'])
        valence = row['valence']
        arousal = row['arousal']

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor([valence, arousal], dtype=torch.float32)
        return image, label

def train_mvitv2(data_dir, output_dir='./regression_mvitv2', num_epochs=20, batch_size=32, lr=5e-4, device='cuda'):
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = VADDataset(
        root_dir=data_dir,
        csv_file=os.path.join(data_dir, 'vad_dataset.csv'),
        transform=transform
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = timm.create_model('mvitv2_small', pretrained=True)
    if isinstance(model.head, nn.Sequential):
        last_layer = model.head[-1]
        if isinstance(last_layer, nn.Linear):
            in_features = last_layer.in_features
        else:
            raise ValueError("Unexpected structure of model head")
    else:
        in_features = model.head.in_features

    model.head = nn.Linear(in_features, 2)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{num_epochs}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(output_dir, f"mvitv2_epoch{epoch+1}.pth"))

    torch.save(model.state_dict(), os.path.join(output_dir, "mvitv2_final.pth"))

    epochs = list(range(1, num_epochs + 1))
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MSE Loss over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    print("Training complete. Models and plots saved.")

if __name__ == "__main__":
    data_dir = r"F:/Users/Ivan/Desktop/vad_dataset"
    output_dir = r"F:/Users/Ivan/Desktop/mvitv2_runs"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_mvitv2(data_dir, output_dir=output_dir, device=device)
