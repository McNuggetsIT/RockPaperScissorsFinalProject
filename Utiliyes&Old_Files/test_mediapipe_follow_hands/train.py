import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import DataLoader, ConcatDataset, random_split

# ================= CONFIG =================
DATASET_DIR = r"Rock-Paper-Scissors"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LR = 0.0002
MODEL_PATH = "model.pth"
CLASSES = ["paper", "rock", "scissors"]  # ordine alfabetico
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# --------- TRANSFORMS ---------
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(35),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.25),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --------- DATASET ---------
train_folder = datasets.ImageFolder(f"{DATASET_DIR}/train", transform=train_tf)
val_folder   = datasets.ImageFolder(f"{DATASET_DIR}/validation", transform=train_tf)

print("Classi trovate:", train_folder.classes)
assert train_folder.classes == CLASSES, "❌ Ordine classi errato!"

# Uniamo train + validation
full_dataset = ConcatDataset([train_folder, val_folder])

# Split RANDOM
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

# --------- MODELLO ---------
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

for p in model.features.parameters():
    p.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.last_channel, 3)
)

model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR)

# --------- TRAINING ---------
best_val = 0.0

for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_acc = correct / total

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1}/{EPOCHS} | Train {train_acc:.3f} | Val {val_acc:.3f}")

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print("✅ modello salvato")

print("Training completato ✔️ | Best Val:", round(best_val, 3))
