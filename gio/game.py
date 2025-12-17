import cv2
import torch
import torch.nn as nn
import random
from torchvision import transforms
import os
import time
from PIL import Image

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
print("SPAZIO = gioca | Q = esci")

# PREPROCESSING
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# FUNZIONI UTILI
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


def draw_text(img, text, pos, scale=1, color=(255,255,255), thickness=2):
    # contorno nero
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0,0,0), thickness+2)
    # testo vero
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness)

# WEBCAM
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Errore apertura webcam")
    exit()

# STATO GIOCO
pc_move = None
result = ""
player_score = 0
pc_score = 0
countdown = 0
countdown_start = None

# LOOP DI GIOCO

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ---------- ROI ----------
    size = 300
    cx, cy = w // 2, h // 2
    roi = frame[
        cy - size // 2 : cy + size // 2,
        cx - size // 2 : cx + size // 2
    ]

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_pil = Image.fromarray(roi_rgb)
    img = transform(roi_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        pred = torch.argmax(outputs, dim=1).item()

    player_move = CLASSES[pred]

    # ---------- INPUT ----------
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" ") and countdown == 0:
        countdown = 3
        countdown_start = time.time()
        pc_move = None
        result = ""

    if key == ord("q"):
        break

    # ---------- COUNTDOWN ----------
    if countdown > 0:
        if time.time() - countdown_start >= 1:
            countdown -= 1
            countdown_start = time.time()

        draw_text(frame, str(countdown + 1),
                  (w//2 - 30, h//2),
                  scale=4, color=(0,0,255), thickness=4)

        if countdown == 0:
            pc_move = random.choice(CLASSES)
            result = decide_winner(player_move, pc_move)

            if result == "HAI VINTO":
                player_score += 1
            elif result == "HAI PERSO":
                pc_score += 1

    # ---------- UI PANEL ----------
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (340, 170), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # ---------- ROI BOX ----------
    cv2.rectangle(frame,
        (cx - size // 2, cy - size // 2),
        (cx + size // 2, cy + size // 2),
        (0, 255, 0), 2
    )

    # ---------- TESTI ----------
    draw_text(frame, f"Tu: {player_move}", (15, 40),
              scale=1, color=(0,255,0))

    if pc_move:
        draw_text(frame, f"PC: {pc_move}", (15, 80),
                  scale=1, color=(255,0,0))

        res_color = (0,255,0) if result == "HAI VINTO" else \
                    (0,0,255) if result == "HAI PERSO" else (0,255,255)

        draw_text(frame, result, (15, 120),
                  scale=1.3, color=res_color, thickness=3)

    draw_text(frame,
        f"Score  Tu {player_score} - {pc_score} PC",
        (15, h - 20),
        scale=0.9,
        color=(255,255,255)
    )

    cv2.imshow("Rock Paper Scissors", frame)

# CLEANUP
cap.release()
cv2.destroyAllWindows()
