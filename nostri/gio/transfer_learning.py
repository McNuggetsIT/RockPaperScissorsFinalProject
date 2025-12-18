import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

# =========================
# PATH DATASET
# =========================
DATA_DIR = Path("data_transfer")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR   = DATA_DIR / "val"
TEST_DIR  = DATA_DIR / "test"

# =========================
# PARAMETRI
# =========================
IMG_SIZE = 30
BATCH_SIZE = 32
EPOCHS = 15
FINE_TUNE_EPOCH = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# TRANSFORM
# =========================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =========================
# DATASET & DATALOADER
# =========================
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_ds   = datasets.ImageFolder(VAL_DIR, transform=val_test_transform)
test_ds  = datasets.ImageFolder(TEST_DIR, transform=val_test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

NUM_CLASSES = len(train_ds.classes)
print("Classi:", train_ds.classes)

# =========================
# MODELLO
# =========================
class RPSNet(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 128, mid_layers: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_in = nn.Linear(64, hidden_size)

        self.mid = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU()
            ) for _ in range(mid_layers)
        ])

        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc_in(x)
        x = self.mid(x)
        return self.fc_out(x)

# =========================
# INIZIALIZZAZIONE
# =========================
model = RPSNet(num_classes=NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# =========================
# TRAINING
# =========================
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # ---- TRAIN ----
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

    # ---- VALIDATION ----
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = torch.argmax(model(images), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Acc: {val_acc:.2f}%")

    # ---- SAVE BEST ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_rps_model_gap.pth")
        print("✔ Miglior modello salvato")

# =========================
# TEST FINALE
# =========================
print("\n🔎 Test finale")

model.load_state_dict(torch.load("best_rps_model_gap.pth", map_location=DEVICE))
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds = torch.argmax(model(images), dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"🎯 Test Accuracy: {100 * correct / total:.2f}%")
