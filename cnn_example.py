'''
Una CNN è composta da più tipi di layer (strati) principali:
Layer Convoluzionale (Convolutional Layer)
    Applica dei filtri (kernels) su piccole porzioni dell’immagine.
    Ogni filtro rileva caratteristiche specifiche: bordi, angoli, texture.
    Risultato: feature map, che mostra dove la caratteristica appare nell’immagine.
Layer di Pooling (Pooling Layer)
    Riduce la dimensione delle feature map (downsampling).
    Tipico: max pooling → prende il valore massimo in una regione.
    Serve a rendere la rete più efficiente e a renderla invariante a piccoli spostamenti.
Layer di Attivazione (Activation Layer)
    Tipicamente ReLU (Rectified Linear Unit).
    Introduce non-linearità, permettendo alla rete di modellare pattern complessi.
Layer Fully Connected (Denso)
    Alla fine della rete, le feature map vengono “appiattite” e collegate a uno strato denso.
    Serve a combinare tutte le informazioni per fare la predizione finale, ad esempio classificare l’immagine
'''

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


# Dataset personalizzato per PyTorch
class NumpyDataset(Dataset):
    # Costruttore della classe: inizializza immagini, label e eventuali trasformazioni
    def __init__(self, images, labels, transform=None):
        self.images = images.astype(np.float32)  # Assicura che le immagini siano float32 (PyTorch lavora meglio con float32)
        self.labels = labels                      # Label numeriche corrispondenti alle classi
        self.transform = transform                # Trasformazioni opzionali (es. data augmentation)

    # Metodo richiesto da PyTorch: restituisce la dimensione del dataset
    def __len__(self):
        return len(self.images)

    # Metodo richiesto da PyTorch: restituisce l'elemento (immagine, label) con indice idx
    def __getitem__(self, idx):
        img = self.images[idx]    # Prende l'immagine corrispondente all'indice
        label = self.labels[idx]  # Prende la label corrispondente

        # Se è stata fornita una trasformazione, la applica
        if self.transform:
            img = self.transform(img)
        else:
            # Se non ci sono trasformazioni, converte in tensore PyTorch
            # PyTorch vuole immagini con canali per primi: [C,H,W] invece di [H,W,C]
            img = torch.from_numpy(img).permute(2,0,1)  

        return img, label   # Restituisce la coppia (immagine, label)
    
class RpsClassifier(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes
    
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

test_dir = "Rock-Paper-Scissors/test"
train_dir = "Rock-Paper-Scissors/train"
val_dir = "Rock-Paper-Scissors/validation"

# Creazione dei dataset usando RpsClassifier
train_dataset = RpsClassifier(train_dir, transform=transform)
val_dataset = RpsClassifier(val_dir, transform=transform)
test_dataset = RpsClassifier(test_dir, transform=transform)

# Creazione dei DataLoader: permettono di iterare sui dataset a batch
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # shuffle=True mischia i dati a ogni epoca
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Definizione CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        # Primo layer convoluzionale: input 3 canali (RGB), output 32 filtri, kernel 3x3
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        # Secondo layer convoluzionale: input 32 filtri, output 64 filtri
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Terzo layer convoluzionale: input 64 filtri, output 128 filtri
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        # Layer di pooling 2x2 (riduce dimensione feature map della metà)
        self.pool = nn.MaxPool2d(2, 2)

        # Layer fully connected: prende tutte le feature map appiattite
        # Dopo 3 convoluzioni + 3 pool su immagini 300x300 → 35x35 dimensione feature map
        self.fc1 = nn.Linear(128*35*35, 128)  # layer hidden con 128 neuroni
        self.fc2 = nn.Linear(128, num_classes) # layer di output con numero di classi

    def forward(self, x):
        # Forward pass: definisce come i dati scorrono nella rete
        x = F.relu(self.conv1(x))  # ReLU dopo la prima convoluzione
        x = self.pool(x)            # pooling
        x = F.relu(self.conv2(x))  # seconda convoluzione + ReLU
        x = self.pool(x)
        x = F.relu(self.conv3(x))  # terza convoluzione + ReLU
        x = self.pool(x)
        x = torch.flatten(x,1)      # appiattisce tutte le feature map in un vettore
        x = F.relu(self.fc1(x))     # fully connected + ReLU
        x = self.fc2(x)             # output layer
        return F.log_softmax(x, dim=1)  # softmax logaritmico per classificazione multi-classe

# Creazione modello, ottimizzatore e funzione di loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # usa GPU se disponibile
model = SimpleCNN(num_classes=3).to(device)  # invia modello al device
optimizer = optim.Adam(model.parameters(), lr=0.001)  # ottimizzatore Adam
criterion = nn.NLLLoss()  # Negative Log Likelihood Loss, compatibile con log_softmax

# Funzione di training
def train(model, loader, optimizer, criterion):
    model.train()  # mette modello in modalità training
    running_loss = 0
    for data, target in loader:  # iterazione sui batch
        data, target = data.to(device), target.to(device)  # invio dati a GPU se disponibile
        optimizer.zero_grad()      # azzera i gradienti
        output = model(data)       # forward pass
        loss = criterion(output, target)  # calcola la loss
        loss.backward()            # backward pass (calcolo gradienti)
        optimizer.step()           # aggiorna i pesi
        running_loss += loss.item()  # accumula la loss
    return running_loss / len(loader)  # ritorna loss media del batch

# Funzione di valutazione (validation/test)
def evaluate(model, loader, criterion):
    model.eval()   # mette modello in modalità eval (disabilita dropout, batchnorm ecc.)
    loss = 0
    correct = 0
    with torch.no_grad():  # disabilita calcolo dei gradienti
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()  # accumula loss
            pred = output.argmax(dim=1)               # prendi la classe con probabilità massima
            correct += (pred == target).sum().item()  # conta quanti corretti
    return loss / len(loader), correct / len(loader.dataset)  # ritorna loss media e accuracy

# Training loop
epochs = 10
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, criterion)  # training su tutti i batch
    val_loss, val_acc = evaluate(model, val_loader, criterion)    # valutazione su validation set
    print(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f} - Val acc: {val_acc:.4f}")

# Test finale
test_loss, test_acc = evaluate(model, test_loader, criterion)  # valutazione finale sul test set
print(f"Test accuracy: {test_acc:.4f}")  # stampa accuracy finale

