import cv2
import os

# Cartella di input con le immagini originali
input_folder = r"test_peppe_mani"

# Cartella di output dove salvare le immagini augmentate
output_folder = os.path.join(input_folder, "augmented")
os.makedirs(output_folder, exist_ok=True)

counter = 1

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Non riesco a leggere {filename}")
            continue

        # Salva copia originale
        out_name = f"Pe_hc_{counter:03d}.jpg"
        cv2.imwrite(os.path.join(output_folder, out_name), img)
        print(f"✔ Salvata copia: {out_name}")
        counter += 1

        # Rotazione leggera
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), 10, 1.0)  # ruota di 10°
        rotated = cv2.warpAffine(img, M, (w, h))
        out_name = f"Pe_hc_{counter:03d}.jpg"
        cv2.imwrite(os.path.join(output_folder, out_name), rotated)
        print(f"✔ Salvata rotazione: {out_name}")
        counter += 1

        # Flip orizzontale
        flipped = cv2.flip(img, 1)
        out_name = f"Pe_hc_{counter:03d}.jpg"
        cv2.imwrite(os.path.join(output_folder, out_name), flipped)
        print(f"✔ Salvato flip: {out_name}")
        counter += 1

        # Aumento luminosità
        brighter = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
        out_name = f"Pe_hc_{counter:03d}.jpg"
        cv2.imwrite(os.path.join(output_folder, out_name), brighter)
        print(f"✔ Salvata versione luminosa: {out_name}")
        counter += 1

        # Riduzione luminosità
        darker = cv2.convertScaleAbs(img, alpha=0.8, beta=-30)
        out_name = f"Pe_hc_{counter:03d}.jpg"
        cv2.imwrite(os.path.join(output_folder, out_name), darker)
        print(f"✔ Salvata versione scura: {out_name}")
        counter += 1
