import os
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import timm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

DATA_DIR = r"F:/Users/Ivan/Desktop/vad_dataset"
CSV_FILE = os.path.join(DATA_DIR, 'vad_dataset.csv')

class VADDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

full_dataset = VADDataset(root_dir=DATA_DIR, csv_file=CSV_FILE, transform=transform)
full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False, num_workers=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = timm.create_model('mvitv2_small', pretrained=True, num_classes=0)
model = model.to(device)
model.eval()

features = []
targets = []

with torch.no_grad():
    for images, labels in tqdm(full_loader, desc="Extracting Features"):
        images = images.to(device)
        feats = model(images).cpu().numpy()
        features.append(feats)
        targets.append(labels.numpy())

features = np.vstack(features)
targets = np.vstack(targets)

X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=42)

valence_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
arousal_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

valence_model.fit(X_train, y_train[:, 0])
arousal_model.fit(X_train, y_train[:, 1])

valence_preds = valence_model.predict(X_val)
arousal_preds = arousal_model.predict(X_val)

print("Valence Metrics:")
print(f"R2: {r2_score(y_val[:, 0], valence_preds):.4f}")
print(f"MAE: {mean_absolute_error(y_val[:, 0], valence_preds):.4f}")
print(f"MSE: {mean_squared_error(y_val[:, 0], valence_preds):.4f}")
print(f"Pearson: {pearsonr(y_val[:, 0], valence_preds)[0]:.4f}")

print("\nArousal Metrics:")
print(f"R2: {r2_score(y_val[:, 1], arousal_preds):.4f}")
print(f"MAE: {mean_absolute_error(y_val[:, 1], arousal_preds):.4f}")
print(f"MSE: {mean_squared_error(y_val[:, 1], arousal_preds):.4f}")
print(f"Pearson: {pearsonr(y_val[:, 1], arousal_preds)[0]:.4f}")
