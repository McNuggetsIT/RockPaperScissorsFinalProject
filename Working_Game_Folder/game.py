import cv2
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torchvision import transforms
import os
import time
from collections import deque
from PIL import Image

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
MODEL_PATH = os.path.join(BASE_DIR, "Working_Game_Folder", "GAG_RPS_Model.pth")

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
    roi_pil = Image.fromarray(roi_rgb)
    img = transform(roi_pil).unsqueeze(0).to(DEVICE)

    # ---------- PREDIZIONE STABILIZZATA ----------
    if countdown == 0:
        with torch.no_grad():
            outputs = model(img)
            probs = F.softmax(outputs, dim=1)[0]
            pred = torch.argmax(probs).item()
            confidence = probs[pred].item()
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
                
            locked_player_move = None
    # =========================
    # OSCURA TUTTO FUORI DALLA ROI
    # =========================

    mask = frame.copy()

    # area sopra la ROI
    cv2.rectangle(mask,
        (0, 0),
        (w, cy - size // 2),
        (0, 0, 0), -1)

    # area sotto la ROI
    cv2.rectangle(mask,
        (0, cy + size // 2),
        (w, h),
        (0, 0, 0), -1)

    # area sinistra della ROI
    cv2.rectangle(mask,
        (0, cy - size // 2),
        (cx - size // 2, cy + size // 2),
        (0, 0, 0), -1)

    # area destra della ROI
    cv2.rectangle(mask,
        (cx + size // 2, cy - size // 2),
        (w, cy + size // 2),
        (0, 0, 0), -1)

    # applica oscuramento (regola alpha se vuoi più/meno buio)
    cv2.addWeighted(mask, 1, frame, 0.35, 0, frame)

    # =========================
    # UI + ROI FISSA (COMPATTA)
    # =========================

    # ---- COLORE BOX IN BASE ALLA GESTURE ----
    box_color = {
        "rock": (0, 0, 255),
        "paper": (0, 255, 0),
        "scissors": (255, 0, 0)
    }.get(player_move, (255, 255, 255))

    # ---- ROI BOX CENTRALE (FISSA) ----
    cv2.rectangle(frame,
        (cx - size // 2, cy - size // 2),
        (cx + size // 2, cy + size // 2),
        box_color, 2
    )

    # ---- PANNELLO UI RIDOTTO (ALTO SINISTRA) ----
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (240, 105), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # ---- TESTI ----
    move_to_show = locked_player_move if locked_player_move else player_move

    draw_text(frame, f"Tu: {move_to_show}",
            (15, 28),
            scale=0.8,
            color=(0,255,0),
            thickness=2)

    if pc_move:
        draw_text(frame, f"PC: {pc_move}",
                (15, 52),
                scale=0.75,
                color=(255,0,0),
                thickness=2)

        res_color = (0,255,0) if result == "HAI VINTO" else \
                    (0,0,255) if result == "HAI PERSO" else (0,255,255)

        draw_text(frame, result,
                (15, 78),
                scale=1.0,
                color=res_color,
                thickness=2)

    # ---- CONFIDENCE BAR (PICCOLA) ----
    bar_x, bar_y = 15, 88
    bar_width = 140
    bar_height = 8

    filled = int(bar_width * confidence)

    cv2.rectangle(frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (70, 70, 70), -1)

    cv2.rectangle(frame,
        (bar_x, bar_y),
        (bar_x + filled, bar_y + bar_height),
        (0, 255, 0), -1)

    draw_text(frame,
        f"{int(confidence * 100)}%",
        (bar_x + bar_width + 8, bar_y + 9),
        scale=0.45,
        color=(200,200,200),
        thickness=1)

    # ---- COMANDI (RIDOTTI, SOTTO IL BOX) ----
    draw_text(frame,
        "SPAZIO = Gioca | Q = Esci",
        (cx - size // 2, cy + size // 2 + 28),
        scale=0.6,
        color=(220,220,220),
        thickness=1)

    # ---- SCORE ----
    draw_text(frame,
        f"Score {player_score} : {pc_score}",
        (15, h - 18),
        scale=0.75,
        color=(220,220,220),
        thickness=1)




    cv2.imshow("Rock Paper Scissors", frame)

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
