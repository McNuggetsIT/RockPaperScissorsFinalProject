import cv2
import torch
import torch.nn as nn
import random
from torchvision import transforms
import os

# PARAMETRI
IMG_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["rock", "paper", "scissors"]

# PATH MODELLO
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "src", "best_rps_model.pth")

# MODELLO 
class RPSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)


# CARICA MODELLO

model = RPSNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))


model.eval()

print("✅ Modello caricato")
print("Premi SPAZIO per giocare | q per uscire")

# PREPROCESSING 
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# FUNZIONE DECIDE VINCITORE
def decide_winner(player, pc):
    if player == pc:
        return "PAREGGIO"
    if (
        (player == "rock" and pc == "scissors") or
        (player == "paper" and pc == "rock") or
        (player == "scissors" and pc == "paper")
    ):
        return "HAI VINTO"
    return "HAI PERSO"

# WEBCAM
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Errore apertura webcam")
    exit()

print("Premi 'q' per uscire")
# stato gioco 
pc_move = None
result = ""
player_score = 0
pc_score = 0
countdown = 0
countdown_start = None

import time

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ROI centrale
    h, w, _ = frame.shape
    size = 300
    cx, cy = w // 2, h // 2
    roi = frame[
        cy - size // 2 : cy + size // 2,
        cx - size // 2 : cx + size // 2
    ]

    # Preprocessing
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = transform(roi_rgb).unsqueeze(0).to(DEVICE)

    # Predizione continua
    with torch.no_grad():
        outputs = model(img)
        pred = torch.argmax(outputs, dim=1).item()

    player_move = CLASSES[pred]

    key = cv2.waitKey(1) & 0xFF

    # AVVIO COUNTDOWN
    if key == ord(" ") and countdown == 0:
        countdown = 3
        countdown_start = time.time()
        pc_move = None
        result = ""

    # GESTIONE COUNTDOWN
    if countdown > 0:
        elapsed = time.time() - countdown_start
        if elapsed >= 1:
            countdown -= 1
            countdown_start = time.time()

        cv2.putText(frame, str(countdown + 1),
                    (w // 2 - 40, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4, (0, 0, 255), 6)

        # FINE COUNTDOWN → GIOCA
        if countdown == 0:
            pc_move = random.choice(CLASSES)
            result = decide_winner(player_move, pc_move)

            if result == "HAI VINTO":
                player_score += 1
            elif result == "HAI PERSO":
                pc_score += 1

    if key == ord("q"):
        break

    # UI
    cv2.rectangle(frame,
        (cx - size // 2, cy - size // 2),
        (cx + size // 2, cy + size // 2),
        (0, 255, 0), 2
    )

    cv2.putText(frame, f"Tu: {player_move}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if pc_move:
        cv2.putText(frame, f"PC: {pc_move}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, result, (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # SCORE
    cv2.putText(frame,
        f"Score  Tu {player_score} - {pc_score} PC",
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (255, 255, 255), 2
    )

    cv2.imshow("Rock Paper Scissors", frame)


# CLEANUP
cap.release()
cv2.destroyAllWindows()