import os
import numpy as np
from PIL import Image

def load_images_from_folders(root_dir, target_size=(300,300)):
    images = []
    labels = []
    class_names = sorted(os.listdir(root_dir))

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(root_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            if file.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(class_path, file)

                img = Image.open(img_path).convert("RGB")
                img = img.resize(target_size)
                img = np.array(img) / 255.0

                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels), class_names

X, y, class_names = load_images_from_folders("Rock-Paper-Scissors/test/")

print("Numero totale immagini:", X.shape[0])
print("Shape immagini:", X.shape) 
print("Shape labels:", y.shape)       
print("Classi:", class_names)
print("Numero classi:", len(class_names))
