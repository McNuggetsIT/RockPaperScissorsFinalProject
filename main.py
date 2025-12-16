import cv2 as cv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm
import PIL as pl

from load_image import load_images_from_folders

test_dir = "Rock-Paper-Scissors/test"
train_dir = "Rock-Paper-Scissors/train"
val_dir = "Rock-Paper-Scissors/validation"

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

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
    
train_dataset = RpsClassifier(
    data_dir=train_dir,
    transform=transform
)    

dataset = RpsClassifier(data_dir=train_dir, transform=transform)
dataset_test = RpsClassifier(data_dir=test_dir, transform=transform)
dataset_validation = RpsClassifier(data_dir=val_dir, transform=transform)

# Visualizza un esempio
image, label = train_dataset[10]

# Denormalizza per mostrare con matplotlib
img = image * 0.5 + 0.5
img = img.permute(1, 2, 0)

plt.imshow(img)
plt.title(train_dataset.classes[label])
plt.axis("off")
plt.show()


target_to_class = {v: k for k, v in ImageFolder(dataset).class_to_idx.items()}
print(target_to_class)
