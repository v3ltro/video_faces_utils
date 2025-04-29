import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_emotions_strong(
    data_dir,
    num_epochs: int = 15,
    batch_size: int = 32,
    lr: float = 2e-4,
    weight_decay: float = 1e-3,
    device: str = 'cuda',
    output_dir: str = './emotieff_runs_b2'
):
    os.makedirs(output_dir, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    num_classes = len(train_dataset.classes)

    # Model
    model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_loss_list, train_acc_list = [], []
    val_acc_list, val_prec_list = [], []
    val_rec_list, val_f1_list   = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Val Epoch {epoch}/{num_epochs}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)

                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc   = 100.0 * correct / total
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1        = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        val_acc_list.append(val_acc)
        val_prec_list.append(precision)
        val_rec_list.append(recall)
        val_f1_list.append(f1)

        print(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss:.4f} | Train Acc={train_acc:.2f}% || "
            f"Val Acc={val_acc:.2f}% | Val Prec={precision:.3f} | "
            f"Val Rec={recall:.3f} | Val F1={f1:.3f}"
        )

        scheduler.step()
        torch.save(model.state_dict(), os.path.join(output_dir, f"b2_epoch{epoch}.pth"))

    torch.save(model.state_dict(), os.path.join(output_dir, "b2_final.pth"))

    epochs = list(range(1, num_epochs + 1))
    def _save_plot(values, label, ylabel, fname):
        plt.figure()
        plt.plot(epochs, values, label=label)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(f'{label} over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, fname))

    _save_plot(train_loss_list, 'Train Loss', 'Loss', 'train_loss.png')
    _save_plot(train_acc_list,  'Train Acc',  'Accuracy (%)', 'train_acc.png')
    _save_plot(val_prec_list,   'Val Precision', 'Precision', 'val_precision.png')
    _save_plot(val_rec_list,    'Val Recall',    'Recall',    'val_recall.png')
    _save_plot(val_f1_list,     'Val F1',        'F1 Score',  'val_f1.png')

    print("Training complete for EfficientNet-B2.")


if __name__ == "__main__":
    DATA_DIR = r"F:/Users/Ivan/Desktop/emotieff_data"
    DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_emotions_strong(
        data_dir=DATA_DIR,
        num_epochs=15,
        batch_size=32,
        lr=2e-4,
        weight_decay=1e-3,
        device=DEVICE,
        output_dir=r"F:/Users/Ivan/Desktop/emotieff_runs_b2_strong"
    )
