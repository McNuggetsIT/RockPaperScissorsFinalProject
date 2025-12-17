import os
import random
import shutil

# Percorso principale del dataset
base_dir = r"Rock-Paper-Scissors"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

# Crea cartelle di validation e test se non esistono
for category in ["rock", "paper", "scissors"]:
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

# Numero minimo e massimo di immagini da spostare
min_images = 5
max_images = 10

for category in ["rock", "paper", "scissors"]:
    train_category_dir = os.path.join(train_dir, category)
    val_category_dir = os.path.join(val_dir, category)
    test_category_dir = os.path.join(test_dir, category)

    # Lista di tutte le immagini .jpg
    images = [f for f in os.listdir(train_category_dir) if f.lower().endswith(".jpg")]

    if len(images) < min_images * 2:
        print(f"⚠ Non ci sono abbastanza immagini in {category} per fare sia validation che test")
        continue

    # Scegli random quante immagini spostare (tra 5 e 10)
    num_val = random.randint(min_images, min(max_images, len(images)//2))
    num_test = random.randint(min_images, min(max_images, len(images)//2))

    # Seleziona immagini random per validation e test
    selected_val = random.sample(images, num_val)
    remaining = [img for img in images if img not in selected_val]
    selected_test = random.sample(remaining, num_test)

    # Sposta immagini in validation
    for img in selected_val:
        src = os.path.join(train_category_dir, img)
        dst = os.path.join(val_category_dir, img)
        shutil.move(src, dst)
        print(f"✔ Spostata {img} da {category}/train a {category}/validation")

    # Sposta immagini in test
    for img in selected_test:
        src = os.path.join(train_category_dir, img)
        dst = os.path.join(test_category_dir, img)
        shutil.move(src, dst)
        print(f"✔ Spostata {img} da {category}/train a {category}/test")

print("✅ Operazione completata!")
