import cv2
import os
import time

# =========================
# PARAMETRI
# =========================
MAX_IMAGES = 200        # immagini per classe
CAPTURE_DELAY = 1.5    # secondi tra uno scatto e l'altro

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "my_dataset", "train")
pers_name = ""

CLASSES = {
    "r": "rock",
    "p": "paper",
    "s": "scissors"
}

# crea cartelle
for cls in CLASSES.values():
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Errore webcam")
    exit()

print("r=rock | p=paper | s=scissors | q=quit")

# contatori
counters = {
    cls: len(os.listdir(os.path.join(DATA_DIR, cls)))
    for cls in CLASSES.values()
}

active_class = None
last_capture_time = 0

# =========================
# LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # UI
    cv2.putText(frame,
        "r=rock  p=paper  s=scissors  q=quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    if active_class:
        cv2.putText(frame,
            f"CAPTURING: {active_class} ({counters[active_class]}/{MAX_IMAGES})",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow("Dataset Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    # uscita
    if key == ord("q"):
        break

    # attiva classe
    if chr(key) in CLASSES:
        active_class = CLASSES[chr(key)]
        print(f"▶ Avvio raccolta: {active_class}")

    # cattura automatica
    if active_class:
        now = time.time()
        if (
            now - last_capture_time >= CAPTURE_DELAY and
            counters[active_class] < MAX_IMAGES
        ):
            counters[active_class] += 1
            filename = f"{active_class}_{counters[active_class]}_{pers_name}.png"
            path = os.path.join(DATA_DIR, active_class, filename)
            cv2.imwrite(path, frame)
            last_capture_time = now
            print(f"✅ Salvata {path}")

        # stop automatico
        if counters[active_class] >= MAX_IMAGES:
            print(f"⏹ Classe {active_class} completata")
            active_class = None

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
