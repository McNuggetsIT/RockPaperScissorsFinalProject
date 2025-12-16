import cv2
import os
import time

# =========================
# DEFINISCO I PATH
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Rock-Paper-Scissors", "testWebCam")

CLASSES = {
    "r": "rock",
    "p": "paper",
    "s": "scissors"
}

# crea le cartelle se non esistono
for cls in CLASSES.values():
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

# WEBCAM  
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Errore apertura webcam")
    exit()

print("Premi:")
print("r = rock | p = paper | s = scissors | q = esci")

# contatori immagini
counters = {cls: len(os.listdir(os.path.join(DATA_DIR, cls)))
            for cls in CLASSES.values()}

# LOOP PER CATTURARE IMMAGINI FINO A CHE NON SI ESCE 
TODO#RACCOLTA IMMAGINI CONTINUATIVE PER 5 MIN PER ARRICCHIRE IL DS

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        "r=rock  p=paper  s=scissors  q=quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if chr(key) in CLASSES:
        label = CLASSES[chr(key)]
        counters[label] += 1

        filename = f"{label}_{counters[label]}.png"
        filepath = os.path.join(DATA_DIR, label, filename)

        cv2.imwrite(filepath, frame)
        print(f"✅ Salvata: {filepath}")

        time.sleep(0.2)  # per evitare le doppie foto

# CLEANUP
cap.release()
cv2.destroyAllWindows()
