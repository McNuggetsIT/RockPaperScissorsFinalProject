import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# PATH DATASET NUOVO
# =========================
train_dir = "data_transfer/train"
val_dir   = "data_transfer/val"
test_dir  = "data_transfer/test"

# =========================
# PARAMETRI
# =========================
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ⚠️ numero classi del NUOVO dataset
NUM_NEW_CLASSES = len(datasets.ImageFolder(train_dir).classes)

# =========================
# TRANSFORM (uguale al vecchio modello)
# =========================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# =========================
# DATASET
# =========================
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset   = datasets.ImageFolder(val_dir, transform=transform)
test_dataset  = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Nuove classi:", train_dataset.classes)

# =========================
# MODELLO (IDENTICO AL VECCHIO)
# =========================
class RPSNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 3)   # 🔹 3 classi
        )

    def forward(self, x):
        return self.net(x)

# =========================
# CARICAMENTO MODELLO PRE-ADDESTRATO
# =========================
model = RPSNet(num_classes=3).to(DEVICE)
model.load_state_dict(torch.load("best_rps_model.pth"))
print("✔ Modello preaddestrato caricato")

# =========================
# FREEZE BACKBONE
# =========================
for param in model.net[:-1].parameters():
    param.requires_grad = False

# =========================
# NUOVO CLASSIFICATORE
# =========================
model.net[-1] = nn.Linear(128, NUM_NEW_CLASSES)
model = model.to(DEVICE)
# =========================
# LOSS & OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.net[-1].parameters(),
    lr=0.001
)

# =========================
# TRAINING + VALIDATION
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
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = torch.argmax(model(images), dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss/len(train_loader):.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_transfer_model.pth")
        print("✔ Miglior modello salvato")

# =========================
# TEST FINALE
# =========================
print("\n🔎 Test finale")

model.load_state_dict(torch.load("best_transfer_model.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds = torch.argmax(model(images), dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"🎯 Test Accuracy: {100*correct/total:.2f}%")
