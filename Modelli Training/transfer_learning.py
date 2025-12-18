import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ==============================
# CONFIG
# ==============================
DATASET_PATH = r"Our_Hand_Dataset"      # dataset/rock paper scissors
MODEL_PATH = "rps_mobilenet.pth"
EPOCHS = 10
BATCH_SIZE = 32
NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# TRANSFORMS (ImageNet)
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# DATASET
# ==============================
dataset = ImageFolder(DATASET_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==============================
# LOAD PRETRAINED MODEL
# (QUI AVVIENE IL DOWNLOAD AUTOMATICO)
# ==============================
model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
)

# Sostituiamo il classificatore finale
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, NUM_CLASSES
)

# Congela il backbone
for param in model.features.parameters():
    param.requires_grad = False

model.to(DEVICE)

# ==============================
# TRAIN SETUP
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

# ==============================
# TRAINING
# ==============================
model.train()
print("🚀 Training started")

for epoch in range(EPOCHS):
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss:.4f}")

# ==============================
# SAVE MODEL
# ==============================

torch.save(model.state_dict(), MODEL_PATH)
print("✅ Training completed")
print("💾 Model saved as:", MODEL_PATH)
