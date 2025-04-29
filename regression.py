import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import numpy as np

class VADDataset(Dataset):
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

def evaluate_regression(model_path, data_dir, output_dir='./regression_runs', batch_size=32, device='cuda'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = VADDataset(
        root_dir=data_dir,
        csv_file=os.path.join(data_dir, 'vad_dataset.csv'),
        transform=transform
    )

    _, val_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images).cpu().numpy()
            labels = labels.numpy()
            preds.append(outputs)
            targets.append(labels)

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    plt.scatter(targets[:, 0], preds[:, 0], alpha=0.5)
    plt.plot([1, 5], [1, 5], '--', color='red')
    plt.xlabel('True Valence')
    plt.ylabel('Predicted Valence')
    plt.title('Valence Prediction')
    plt.savefig(os.path.join(output_dir, 'scatter_valence.png'))
    plt.close()

    plt.figure()
    plt.scatter(targets[:, 1], preds[:, 1], alpha=0.5)
    plt.plot([1, 5], [1, 5], '--', color='red')
    plt.xlabel('True Arousal')
    plt.ylabel('Predicted Arousal')
    plt.title('Arousal Prediction')
    plt.savefig(os.path.join(output_dir, 'scatter_arousal.png'))
    plt.close()

    print("Evaluation complete. Scatter plots saved.")

if __name__ == "__main__":
    data_dir = r"F:/Users/Ivan/Desktop/vad_dataset"
    model_path = r"F:/Users/Ivan/Desktop/regression_runs/resnet18_final.pth"
    output_dir = r"F:/Users/Ivan/Desktop/regression_runs"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    evaluate_regression(model_path, data_dir, output_dir=output_dir, device=device)
