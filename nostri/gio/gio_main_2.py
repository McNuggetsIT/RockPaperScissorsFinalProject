import random
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
from torchvision import datasets
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from sklearn.metrics import classification_report, confusion_matrix
import os
import warnings
warnings.filterwarnings("ignore")

# =========================
# PATH DATASET
# =========================
train_dir = "Rock-Paper-Scissors/train"
val_dir   = "Rock-Paper-Scissors/validation"
test_dir  = "Rock-Paper-Scissors/test"

# =========================
# TRANSFORM
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

validation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])



# =========================
# DATASET
# =========================
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset  = datasets.ImageFolder(test_dir, transform=test_transform)

print("Classi:", train_dataset.classes)
print("Train:", len(train_dataset))
print("Test:", len(test_dataset))

# =========================
# DATALOADER
# =========================
BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# =========================
# MODELLO CNN
# =========================
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 5)
        
        self.fc1 = nn.Linear(256, 50)
        
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.fc1(x)
        return x

# =========================
# INIZIALIZZAZIONE
# =========================
lr = 1e-3
epochs = 20
device = ('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel().to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# =========================
# TRAINING
# =========================
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

# Initialize a variable to store the state dictionary of the best model
best_model_state = None
best_valid_acc = 0.0

train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                              optimizer, criterion)
    valid_epoch_loss, valid_epoch_acc = validate(model, test_loader,  
                                                 criterion)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    print('-'*50)
    # Check if the current model is the best one based on validation accuracy
    if valid_epoch_acc > best_valid_acc:
        best_valid_acc = valid_epoch_acc
        best_model_state = model.state_dict().copy()  # Deep copy the model state
        print(f"Updated best model in memory with validation accuracy: {valid_epoch_acc:.3f}")
        torch.save(model.state_dict(), "best_rps_model.pth")
        print("✔ Modello salvato")
    time.sleep(5)
    
def show_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    plt.style.use('ggplot')
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
show_plots(train_acc, valid_acc, train_loss, valid_loss)

class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.images = os.listdir(data_folder)
        self.classes = ['rock', 'paper', 'scissors']
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.data_folder, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        label = next((c for c in self.classes if c in self.images[idx]), None)
        if self.transform:
            image = self.transform(image)
        return image, label
    
validation_dataset = CustomDataset(val_dir, transform=validation_transform)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

for images, labels in validation_loader:
    idx = random.randint(0, images.size(0) - 1)
    image = images[idx]
    image = image.permute(1, 2, 0).numpy()
    image = (image * 0.5) + 0.5
    plt.imshow(image)
    plt.title(f'Class label: {labels[idx]}')
    plt.axis('off')
    plt.show()
    break

final_model = CNNModel()
final_model.load_state_dict(best_model_state)
final_model.eval()

predictions = []
real_values = []
with torch.no_grad():
    for idx, data in enumerate(validation_loader):
        images, labels = data
        images = images.to(device)
        real_values.extend(labels)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.tolist())
        
predictions = [train_dataset.classes[idx] for idx in predictions]
print(classification_report(real_values, predictions))