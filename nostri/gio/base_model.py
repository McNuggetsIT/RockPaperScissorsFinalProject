import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# =========================
# PATH DATASET
# =========================
train_dir = "Rock-Paper-Scissors/train"
val_dir   = "Rock-Paper-Scissors/validation"
test_dir  = "Rock-Paper-Scissors/test"

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((140, 140)),
    transforms.RandomCrop((128, 128)),  # 🔥 rompe il bias di altezza
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])



# =========================
# DATASET
# =========================
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset   = datasets.ImageFolder(val_dir, transform=transform)
test_dataset  = datasets.ImageFolder(test_dir, transform=transform)

print("Classi:", train_dataset.classes)
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

# =========================
# DATALOADER
# =========================
BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# DEVICE
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# =========================
# MODELLO CNN
# =========================
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
            nn.Linear(64 * 30 * 30, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)

model = RPSNet().to(DEVICE)

# =========================
# LOSS & OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# TRAINING
# =========================
EPOCHS = 10
patience = 3

train_losses = []
val_accuracies = []

best_val_acc = 0.0
epochs_no_improve = 0

for epoch in range(EPOCHS):

    # ---- TRAIN ----
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # ---- VALIDATION ----
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total

    train_losses.append(avg_train_loss)
    val_accuracies.append(val_acc)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {avg_train_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    # ---- EARLY STOPPING ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_rps_model.pth")
        print("✔ Modello salvato")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("⏹ Early stopping")
            break

# =========================
# GRAFICI
# =========================
epochs_range = range(1, len(train_losses) + 1)

plt.figure()
plt.plot(epochs_range, train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss")
plt.grid()
plt.show()

plt.figure()
plt.plot(epochs_range, val_accuracies, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Validation Accuracy")
plt.grid()
plt.show()

# =========================
# TEST + CONFUSION MATRIX
# =========================
model.load_state_dict(torch.load("best_rps_model.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=train_dataset.classes,
    yticklabels=train_dataset.classes
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# =========================
# ACCURACY PER CLASSE
# =========================
class_correct = [0] * 3
class_total = [0] * 3

for t, p in zip(all_labels, all_preds):
    class_correct[t] += int(t == p)
    class_total[t] += 1

class_acc = [100 * class_correct[i] / class_total[i] for i in range(3)]

plt.figure()
plt.bar(train_dataset.classes, class_acc)
plt.ylabel("Accuracy (%)")
plt.title("Accuracy per classe")
plt.show()

# =========================
# ERRORI VISUALI
# =========================
images, labels = next(iter(test_loader))
images = images.to(DEVICE)

outputs = model(images)
preds = torch.argmax(outputs, dim=1)

wrong_idx = (preds != labels.to(DEVICE)).nonzero(as_tuple=True)[0][:5]

for idx in wrong_idx:
    img = images[idx].cpu().permute(1, 2, 0)
    plt.figure()
    plt.imshow(img)
    plt.title(
        f"True: {train_dataset.classes[labels[idx]]} | "
        f"Pred: {train_dataset.classes[preds[idx]]}"
    )
    plt.axis("off")
    plt.show()
