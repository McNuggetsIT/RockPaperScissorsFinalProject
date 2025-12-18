import cv2
import mediapipe as mp
import os
import time

# ============r===== CONFIG =================
DATASET_DIR = r"Rock-Paper-Scissors"
CLASSES = ["paper", "rock", "scissors"]
IMAGES_PER_CLASS = 40
DELAY_BETWEEN_SHOTS = 0.25  # secondi
# =========================================

os.makedirs(DATASET_DIR, exist_ok=True)
for c in CLASSES:
    os.makedirs(os.path.join(DATASET_DIR, c), exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

def capture_class(class_name):
    print(f"\n👉 PREPARATI: {class_name.upper()}")
    print("   - mano ben visibile")
    print("   - gesto CHIARO")
    print("   - tieni fermo per 1s\n")
    time.sleep(3)

    count = 0
    last_shot = time.time()

    while count < IMAGES_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            hand = results.multi_hand_landmarks[0]
            xs = [int(lm.x * w) for lm in hand.landmark]
            ys = [int(lm.y * h) for lm in hand.landmark]

            x1, x2 = max(min(xs)-30, 0), min(max(xs)+30, w)
            y1, y2 = max(min(ys)-30, 0), min(max(ys)+30, h)

            roi = frame[y1:y2, x1:x2]

            if roi.size > 0 and time.time() - last_shot > DELAY_BETWEEN_SHOTS:
                path = os.path.join(DATASET_DIR, class_name, f"{count}.jpg")
                cv2.imwrite(path, roi)
                count += 1
                last_shot = time.time()
                print(f"{class_name}: {count}/{IMAGES_PER_CLASS}")

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        cv2.putText(
            frame,
            f"{class_name.upper()}  {count}/{IMAGES_PER_CLASS}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow("Dataset Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

for cls in CLASSES:
    capture_class(cls)

cap.release()
cv2.destroyAllWindows()

print("\n✅ DATASET COMPLETATO")
print("Ora puoi allenare il modello con queste immagini.")
