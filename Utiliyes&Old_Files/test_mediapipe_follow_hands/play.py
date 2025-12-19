import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import mediapipe as mp
import numpy as np
import random
from collections import deque, Counter

# ================= CONFIG =================
IMG_SIZE = 224
MODEL_PATH = "model.pth"
CLASSES = ["paper", "rock", "scissors"]
CONF_THRESHOLD = 0.55
STABLE_FRAMES = 6
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- MODELLO --------
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 3)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -------- MEDIAPIPE --------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
history = deque(maxlen=STABLE_FRAMES)

def winner(p, c):
    if p == c: return "DRAW"
    if (p == "rock" and c == "scissors") or \
       (p == "scissors" and c == "paper") or \
       (p == "paper" and c == "rock"):
        return "YOU WIN"
    return "CPU WINS"

print("SPACE = gioca | Q = esci")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    final_pred = "unknown"
    conf = 0.0

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        h, w, _ = frame.shape
        xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
        ys = [int(lm.y * h) for lm in hand_landmarks.landmark]

        x1, x2 = max(min(xs)-30, 0), min(max(xs)+30, w)
        y1, y2 = max(min(ys)-30, 0), min(max(ys)+30, h)

        # ---- RITAGLIO MANO ----
    roi = frame[y1:y2, x1:x2]

    # maschera semplice per ridurre foglio bianco
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    roi = cv2.bitwise_and(roi, roi, mask=mask)


    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.putText(frame, f"Hand: {final_pred} ({conf:.2f})", (20,40),
    cv2.FONT_HERSHEY_SIMPLEX, 1,
    (0,255,0) if final_pred!="unknown" else (0,0,255), 2)

    cv2.imshow("Rock Paper Scissors", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == 32 and final_pred != "unknown":
        cpu = random.choice(CLASSES)
        print(f"You: {final_pred} | CPU: {cpu} -> {winner(final_pred, cpu)}")
        history.clear()

cap.release()
cv2.destroyAllWindows()
