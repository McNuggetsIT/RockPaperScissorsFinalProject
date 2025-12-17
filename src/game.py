import cv2
import torch
import torch.nn as nn
import random
from torchvision import transforms
import os
import time
from collections import deque

# =========================
# PARAMETRI
# =========================
IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["rock", "paper", "scissors"]

# =========================
# PATH MODELLO
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "src", "best_rps_model.pth")

# =========================
# MODELLO
# =========================
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
            nn.Linear(64 * 30 * 30, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# CARICA MODELLO
# =========================
model = RPSNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =========================
# PREPROCESSING (UGUALE AL TRAINING)
# =========================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# =========================
# FUNZIONI UTILI
# =========================
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
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0,0,0), thickness+2)
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness)

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Errore apertura webcam")
    exit()

# =========================
# STATO GIOCO
# =========================
pc_move = None
result = ""
player_score = 0
pc_score = 0
countdown = 0
countdown_start = None
locked_player_move = None
player_move = "?"

# BUFFER PER STABILIZZAZIONE
pred_buffer = deque(maxlen=5)

# =========================
# LOOP DI GIOCO
# =========================
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
# --- PREPROCESSING ROBUSTO PER WEBCAM ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Debug (lascialo acceso la prima volta)
    cv2.imshow("HAND MASK", thresh)

    roi_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    img = transform(roi_rgb).unsqueeze(0).to(DEVICE)


    img = transform(roi_rgb).unsqueeze(0).to(DEVICE)

    # ---------- PREDIZIONE STABILIZZATA ----------
    if countdown == 0:
        with torch.no_grad():
            outputs = model(img)
            pred = torch.argmax(outputs, dim=1).item()
            raw_move = CLASSES[pred]

        pred_buffer.append(raw_move)
        player_move = max(set(pred_buffer), key=pred_buffer.count)

    # ---------- INPUT ----------
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" ") and countdown == 0:
        countdown = 3
        countdown_start = time.time()
        pc_move = None
        result = ""
        locked_player_move = player_move   # 🔒 blocca subito
        pred_buffer.clear()                # reset buffer

    if key == ord("q"):
        break

    # ---------- COUNTDOWN ----------
    if countdown > 0:
        if time.time() - countdown_start >= 1:
            countdown -= 1
            countdown_start = time.time()

        draw_text(frame, str(countdown),
                  (w//2 - 30, h//2),
                  scale=4, color=(0,0,255), thickness=4)

        if countdown == 0:
            pc_move = random.choice(CLASSES)
            result = decide_winner(locked_player_move, pc_move)

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
    move_to_show = locked_player_move if locked_player_move else player_move
    draw_text(frame, f"Tu: {move_to_show}", (15, 40),
              scale=1, color=(0,255,0))

    draw_text(frame,
        "SPAZIO = Gioca | Q = Esci",
        (cx - size//2, cy + size//2 + 40),
        scale=0.8, color=(255,255,255))

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
        scale=0.9, color=(255,255,255))

    cv2.imshow("Rock Paper Scissors", frame)

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
