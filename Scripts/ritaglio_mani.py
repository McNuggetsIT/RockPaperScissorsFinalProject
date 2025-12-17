import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Cartella con le immagini
folder = r"E:\Repo Machine\RockPaperScissorsFinalProject\test_peppe_mani"

# Percorso del modello .task (scaricalo da Google MediaPipe e mettilo nella cartella del progetto)
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
model_path = r"E:\Repo Machine\RockPaperScissorsFinalProject\hand_landmarker.task"

# Configura il rilevatore
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Contatore per i nomi dei file
counter = 1

# Cicla tutte le immagini .jpg nella cartella
for filename in os.listdir(folder):
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Non riesco a leggere {filename}")
            continue

        mp_image = mp.Image.create_from_file(img_path)
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            h, w, _ = img.shape
            xs, ys = [], []
            for lm in result.hand_landmarks[0]:
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # Padding più ampio per includere sfondo
            pad_x = 50
            pad_y = 50
            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(w, x_max + pad_x)
            y_max = min(h, y_max + pad_y)

            hand_crop = img[y_min:y_max, x_min:x_max]

            # Nome file sequenziale: Pe_hc_001, Pe_hc_002, ...
            out_name = f"Pe_hc_{counter:03d}.jpg"
            out_path = os.path.join(folder, out_name)
            cv2.imwrite(out_path, hand_crop)
            print(f"✔ Mano ritagliata salvata: {out_name}")

            counter += 1
        else:
            print(f"❌ Nessuna mano rilevata in {filename}")
