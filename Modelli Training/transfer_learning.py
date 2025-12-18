import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# PATH DATASET
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
FINE_TUNE_EPOCH = 5   # epoca in cui inizia il fine tuning

# =========================
# TRANSFORM
# Usa 64x64, compatibile con tuo nuovo modello
# =========================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# =========================
# DATASET & DATALOADER
# =========================
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset   = datasets.ImageFolder(val_dir, transform=transform)
test_dataset  = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

NUM_NEW_CLASSES = len(train_dataset.classes)
print("Nuove classi:", train_dataset.classes)

# =========================
# MODELLO
# =========================
class RPSNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3),       # 0
            nn.ReLU(),                 # 1
            nn.MaxPool2d(2),           # 2

            nn.Conv2d(32, 64, 3),      # 3
            nn.ReLU(),                 # 4
            nn.MaxPool2d(2),           # 5

            nn.Flatten(),              # 6
            nn.Linear(64*14*14, 128),  # 7
            nn.ReLU(),                 # 8
            nn.Linear(128, num_classes) # 9
        )

    def forward(self, x):
        return self.net(x)

# =========================
# CARICAMENTO MODELLO PRE-ADDESTRATO
# =========================
model = RPSNet(num_classes=3).to(DEVICE)

# Carico SOLO layer compatibili per evitare mismatch
pretrained_dict = torch.load("best_rps_model.pth", map_location=DEVICE)
model_dict = model.state_dict()

# filtra i pesi compatibili (con stessa shape)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

print("✔ Modello pre-addestrato caricato (pesi compatibili)")

# =========================
# NUOVO CLASSIFICATORE
# =========================
model.net[9] = nn.Linear(128, NUM_NEW_CLASSES).to(DEVICE)

# =========================
# FASE 1 — FREEZE BACKBONE
# =========================
for param in model.net[:6].parameters():  # conv + pool
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.net[9].parameters(), lr=1e-3)

# =========================
# TRAINING + FINE TUNING
# =========================
best_val_acc = 0.0

for epoch in range(EPOCHS):

    # 🔓 Inizio FINE TUNING
    if epoch == FINE_TUNE_EPOCH:
        print("\n🔓 Avvio FINE TUNING (sblocco Conv2)")

        for param in model.net[3:6].parameters():  # Conv2 + ReLU + Pool
            param.requires_grad = True

        optimizer = optim.Adam([
            {"params": model.net[3:6].parameters(), "lr": 1e-4},
            {"params": model.net[9].parameters(), "lr": 1e-3}
        ])

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

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {train_loss/len(train_loader):.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_finetuned_model.pth")
        print("✔ Miglior modello salvato")

# =========================
# TEST FINALE
# =========================
print("\n🔎 Test finale")

model.load_state_dict(torch.load("best_finetuned_model_funzionante.pth", map_location=DEVICE))
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds = torch.argmax(model(images), dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"🎯 Test Accuracy: {100*correct/total:.2f}%")
