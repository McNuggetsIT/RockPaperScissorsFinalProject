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
    
dataset = RpsClassifier(data_dir= train_dir)

image, label = dataset[10]

plt.imshow(image)
plt.title(dataset.classes[label])
plt.axis("off")
plt.show()
