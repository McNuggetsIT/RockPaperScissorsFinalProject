import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

test_dir = "Rock-Paper-Scissors/test"
train_dir = "Rock-Paper-Scissors/train"
val_dir = "Rock-Paper-Scissors/validation"

transform = transforms.Compose([
    transforms.Resize((64, 64)),   
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(train_dir, transform=transform)

print("Classi:", dataset.classes)
print("Numero immagini:", len(dataset))

#controllo dimensioni immagini
img, label = dataset[0]
print("Dimensione immagine:", img.shape)

val_ratio = 0.2
val_size = int(val_ratio * len(dataset))
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size]
)

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class RPSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)

model = RPSNet().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10
patience = 3

best_val_acc = 0.0
epochs_without_improvement = 0
for epoch in range(EPOCHS):

    # ===== TRAINING =====
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ===== VALIDATION =====
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.3f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    # ===== EARLY STOPPING + SALVATAGGIO =====
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_rps_model.pth")
        print("Nuovo modello migliore salvato!")
    else:
        epochs_without_improvement += 1
        print(f"Nessun miglioramento ({epochs_without_improvement}/{patience})")

    if epochs_without_improvement >= patience:
        print("Early stopping attivato")
        break

