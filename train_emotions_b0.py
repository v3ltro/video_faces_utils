import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_efficientnet_b0(data_dir, num_epochs=15, batch_size=32, lr=0.001, device='cuda',
                          output_dir='./emotieff_runs_b0'):
    os.makedirs(output_dir, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'),
                                         transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = len(train_dataset.classes)

    # Model
    model = efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_acc_list = []
    val_acc_list = []
    val_precision_list = []
    val_recall_list = []
    val_f1_list = []
    train_loss_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader,
                                       desc=f"Validating Epoch {epoch + 1}/{num_epochs}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = 100. * correct / total
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        val_acc_list.append(val_acc)
        val_precision_list.append(precision)
        val_recall_list.append(recall)
        val_f1_list.append(f1)

        print(
            f"Epoch {epoch + 1}: Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}% | Val Precision={precision:.4f} | Val Recall={recall:.4f} | Val F1={f1:.4f}")

        scheduler.step()

        torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch{epoch + 1}.pth"))

    torch.save(model.state_dict(), os.path.join(output_dir, "model_final.pth"))

    epochs = list(range(1, num_epochs + 1))

    plt.figure()
    plt.plot(epochs, train_loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'train_loss.png'))

    plt.figure()
    plt.plot(epochs, train_acc_list, label='Train Accuracy')
    plt.plot(epochs, val_acc_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy.png'))

    plt.figure()
    plt.plot(epochs, val_precision_list, label='Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Validation Precision over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'val_precision.png'))

    plt.figure()
    plt.plot(epochs, val_recall_list, label='Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Validation Recall over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'val_recall.png'))

    plt.figure()
    plt.plot(epochs, val_f1_list, label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'val_f1.png'))

    print("Training complete for EfficientNet-b0.")


if __name__ == "__main__":
    data_dir = r"F:/Users/Ivan/Desktop/emotieff_data"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_efficientnet_b0(data_dir=data_dir, num_epochs=15, batch_size=32, device=device)

