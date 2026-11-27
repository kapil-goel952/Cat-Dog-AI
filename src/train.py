
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from pathlib import Path
from tqdm import tqdm

from model import CatDogCNN


# Configuration

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "dataset" / "validation"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 12

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

torch.manual_seed(42)
torch.set_num_threads(4)

device = torch.device("cpu")

print("Training device:", device)


# Image transformations

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
    ),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


# Load datasets

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise ValueError("Training and validation class mappings differ!")

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError("Training or validation dataset is empty!")

print("\nClasses:", train_dataset.class_to_idx)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# Create pretrained model

model = CatDogCNN(pretrained=True).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam([
    {
        "params": model.backbone.layer4.parameters(),
        "lr": 0.0001
    },
    {
        "params": model.backbone.fc.parameters(),
        "lr": 0.001
    }
], weight_decay=0.0001)


# Training loop

best_accuracy = 0.0

for epoch in range(EPOCHS):

    model.train()

    # Keep earlier frozen BatchNorm layers in evaluation mode.
    model.backbone.bn1.eval()
    model.backbone.layer1.eval()
    model.backbone.layer2.eval()
    model.backbone.layer3.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, labels in progress:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

        progress.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = running_loss / total
    train_accuracy = 100 * correct / total


    # Validation

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss_total = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            val_total += labels.size(0)
            val_correct += (predictions == labels).sum().item()

    val_loss = val_loss_total / val_total

    # Important: calculate validation accuracy.
    val_accuracy = 100 * val_correct / val_total

    print(f"\nEpoch: {epoch + 1}/{EPOCHS}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")


    # Save the best model separately.
    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_to_idx": train_dataset.class_to_idx,
            "image_size": IMAGE_SIZE,
            "mean": MEAN,
            "std": STD
        }, MODEL_DIR / "catdog_resnet18_best.pth")

        print("Best model saved!")


print("\nTraining completed!")
print(f"Best validation accuracy: {best_accuracy:.2f}%")
