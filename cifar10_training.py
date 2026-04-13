import argparse
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 64
EPOCHS     = 15
LR         = 0.001
CLASSES    = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def compute_mean_std():
    raw = torchvision.datasets.CIFAR10(root='.', train=True, download=False,
                                       transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(raw, batch_size=len(raw), num_workers=2)
    images, _ = next(iter(loader))
    mean = tuple(images.mean(dim=[0, 2, 3]).tolist())
    std  = tuple(images.std(dim=[0, 2, 3]).tolist())
    print(f'MEAN: {tuple(round(v, 4) for v in mean)}')
    print(f'STD:  {tuple(round(v, 4) for v in std)}')
    return mean, std

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def build_loaders(mean, std):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='.', train=True,  download=False, transform=train_transform)
    test_dataset  = torchvision.datasets.CIFAR10(root='.', train=False, download=False, transform=test_transform)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader   = torch.utils.data.DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f'Train samples: {len(train_dataset):,}  |  Test samples: {len(test_dataset):,}')
    return train_loader, test_loader

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, name, train_loader, epochs=EPOCHS):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    history = {'loss': []}

    print(f'\n=== Training {name} for {epochs} epochs ===')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{epochs}]', leave=True)
        for inputs, labels in progress:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress.set_postfix(loss=f'{running_loss / (progress.n + 1):.4f}')
        history['loss'].append(running_loss / len(train_loader))

    print(f'{name} training complete.')
    return model, history

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            preds  = model(inputs).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    accuracy   = (all_preds == all_labels).mean()
    n          = len(all_labels)
    ci         = 1.96 * np.sqrt(accuracy * (1 - accuracy) / n)
    return accuracy, ci, all_preds, all_labels

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confidence_intervals(simple_acc, simple_ci, deep_acc, deep_ci):
    models     = ['SimpleCNN', 'DeepCNN']
    accuracies = [simple_acc, deep_acc]
    cis        = [simple_ci,  deep_ci]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(models, [a * 100 for a in accuracies], yerr=[c * 100 for c in cis],
                  capsize=8, color=['steelblue', 'darkorange'], alpha=0.85, width=0.4)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Model Accuracy with 95% Confidence Intervals')
    ax.set_ylim(0, 100)
    for bar, acc, ci in zip(bars, accuracies, cis):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci * 100 + 1.5,
                f'{acc:.2%}\n±{ci:.4f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('confidence_intervals.png', dpi=150)
    plt.close()


def plot_training_curves(simple_history, deep_history):
    plt.figure(figsize=(8, 4))
    plt.plot(simple_history['loss'], label='SimpleCNN', marker='o')
    plt.plot(deep_history['loss'],   label='DeepCNN',   marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.close()


def plot_confusion_matrices(simple_acc, simple_ci, simple_preds, simple_labels,
                            deep_acc,   deep_ci,   deep_preds,   deep_labels):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, preds, labels, title in [
        (axes[0], simple_preds, simple_labels, f'SimpleCNN  ({simple_acc:.2%} ± {simple_ci:.4f})'),
        (axes[1], deep_preds,   deep_labels,   f'DeepCNN    ({deep_acc:.2%} ± {deep_ci:.4f})'),
    ]:
        cm   = confusion_matrix(labels, preds, normalize='true')
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
        disp.plot(ax=ax, colorbar=True, xticks_rotation=45, values_format='.2f')
        ax.set_title(title)
    plt.suptitle('Normalized Confusion Matrices (row = true class)', fontsize=13)
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=150)
    plt.close()


def print_per_class_accuracy(simple_preds, simple_labels, deep_preds, deep_labels):
    print(f'\n{"Class":<12} {"SimpleCNN":>10}  {"DeepCNN":>10}')
    print('-' * 36)
    for i, cls in enumerate(CLASSES):
        mask  = simple_labels == i
        s_acc = (simple_preds[mask] == i).mean()
        d_acc = (deep_preds[mask]   == i).mean()
        print(f'{cls:<12} {s_acc:>9.2%}  {d_acc:>9.2%}')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 CNN training and evaluation')
    parser.add_argument('--train', action='store_true',
                        help='Train models from scratch (default: load saved weights)')
    args = parser.parse_args()

    print(f'Using device: {device}')

    mean, std = compute_mean_std()
    train_loader, test_loader = build_loaders(mean, std)

    print(f'\nSimpleCNN parameters: {sum(p.numel() for p in SimpleCNN().parameters()):,}')
    print(f'DeepCNN  parameters: {sum(p.numel() for p in DeepCNN().parameters()):,}')

    if args.train:
        simple_model, simple_history = train_model(SimpleCNN(), 'SimpleCNN', train_loader)
        deep_model,   deep_history   = train_model(DeepCNN(),   'DeepCNN',   train_loader)
        torch.save(simple_model.state_dict(), 'simple_cnn.pth')
        torch.save(deep_model.state_dict(),   'deep_cnn.pth')
        print('\nModels saved to simple_cnn.pth and deep_cnn.pth')
    else:
        import os
        missing = [f for f in ('simple_cnn.pth', 'deep_cnn.pth') if not os.path.exists(f)]
        if missing:
            print(f'Error: weight file(s) not found: {", ".join(missing)}')
            print('Run with --train to train from scratch first.')
            sys.exit(1)
        simple_model = SimpleCNN().to(device)
        simple_model.load_state_dict(torch.load('simple_cnn.pth', map_location=device))
        deep_model   = DeepCNN().to(device)
        deep_model.load_state_dict(torch.load('deep_cnn.pth',   map_location=device))
        print('Models loaded from simple_cnn.pth and deep_cnn.pth')

    # Evaluation
    simple_acc, simple_ci, simple_preds, simple_labels = evaluate(simple_model, test_loader)
    deep_acc,   deep_ci,   deep_preds,   deep_labels   = evaluate(deep_model,   test_loader)

    print(f'\n{"Model":<12} {"Accuracy":>10}  {"95% CI":>14}')
    print('-' * 40)
    print(f'{"SimpleCNN":<12} {simple_acc:>9.2%}  ±{simple_ci:.4f}')
    print(f'{"DeepCNN":<12} {deep_acc:>9.2%}  ±{deep_ci:.4f}')

    # Plots
    plot_confidence_intervals(simple_acc, simple_ci, deep_acc, deep_ci)

    if args.train:
        plot_training_curves(simple_history, deep_history)

    plot_confusion_matrices(simple_acc, simple_ci, simple_preds, simple_labels,
                            deep_acc,   deep_ci,   deep_preds,   deep_labels)
    print_per_class_accuracy(simple_preds, simple_labels, deep_preds, deep_labels)


if __name__ == '__main__':
    # sys.argv = ['cifar10_training.py', '--train']
    sys.argv = ['cifar10_training.py']  # run this if you already have trained weights
    main()
