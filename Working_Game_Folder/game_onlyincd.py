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
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Working_Game_Folder", "GAG_RPS_Model.pth")
BG_PATH = os.path.join(BASE_DIR, "Working_Game_Folder", "Bg23.png")
SPLASH_PATH = os.path.join(BASE_DIR, "Working_Game_Folder", "splash_screen.png")

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
# FUNZIONI
# =========================
def decide_winner(player, pc):
    if player == pc:
        return "PAREGGIO"
    if (player == "rock" and pc == "scissors") or \
       (player == "paper" and pc == "rock") or \
       (player == "scissors" and pc == "paper"):
        return "HAI VINTO"
    return "HAI PERSO"

def draw_text(img, text, pos, scale=1, color=(255,255,255), thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0,0,0), thickness+2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness)

# =========================
# CARICAMENTO MODELLO
# =========================
model = RPSNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =========================
# SPLASH SCREEN
# =========================
splash = cv2.imread(SPLASH_PATH)
if splash is not None:
    splash = cv2.resize(splash, (1280, 720))
    cv2.imshow("Rock Paper Scissors", splash)
    start = time.time()
    while time.time() - start < 2.5:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyWindow("Rock Paper Scissors")

# =========================
# CARICAMENTO ASSET
# =========================
bg_img = cv2.imread(BG_PATH)
if bg_img is None:
    raise FileNotFoundError(f"❌ Background mancante: {BG_PATH}")

# =========================
# FINESTRA
# =========================
WINDOW_NAME = "Rock Paper Scissors"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 800)

# =========================
# PREPROCESSING
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise RuntimeError("❌ Errore apertura webcam")

# =========================
# STATO
# =========================
player_score = 0
pc_score = 0
player_move = "?"
pc_move = None
result = ""
confidence = 0.0
countdown = 0
countdown_start = None
locked_player_move = None
pred_buffer = deque(maxlen=5)
reset_timer = None  # timer per reset automatico dopo il round

# =========================
# ROI PER MODELLO
# =========================
ROI_W = 340
ROI_H = 340

# =========================
# FIT CAM NEL MONITOR (solo grafica)
# =========================
SCREEN_W = 360
SCREEN_H = 260
SCREEN_X_OFF = 0
SCREEN_Y = 140

# =========================
# LOOP PRINCIPALE
# =========================
while True:
    ret, cam = cap.read()
    if not ret:
        break

    cam = cv2.flip(cam, 1)
    h, w, _ = cam.shape

    # =========================
    # ROI WEBCAM
    # =========================
    cy, cx = h // 2, w // 2
    y1 = max(0, cy - ROI_H // 2)
    y2 = min(h, cy + ROI_H // 2)
    x1 = max(0, cx - ROI_W // 2)
    x2 = min(w, cx + ROI_W // 2)
    roi = cam[y1:y2, x1:x2]
    roi = cv2.resize(roi, (ROI_W, ROI_H))

    # =========================
    # PREPROCESSING
    # =========================
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img = transform(Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))).unsqueeze(0).to(DEVICE)

    # =========================
    # PREDIZIONE SOLO DURANTE COUNTDOWN
    # =========================
    if countdown > 0:
        with torch.no_grad():
            probs = F.softmax(model(img), dim=1)[0]
            pred = torch.argmax(probs).item()
            confidence = probs[pred].item()
            pred_buffer.append(CLASSES[pred])
            player_move = max(set(pred_buffer), key=pred_buffer.count)
    else:
        if reset_timer is None:  # fuori dal countdown e prima del reset
            player_move = "?"
            confidence = 0.0
            pred_buffer.clear()

    # =========================
    # INPUT TASTIERA
    # =========================
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" ") and countdown == 0:
        countdown = 3
        countdown_start = time.time()
        pc_move = None
        result = ""
        locked_player_move = None
        pred_buffer.clear()
        reset_timer = None
        player_move = "?"

    if key == ord("q"):
        break

    # =========================
    # COUNTDOWN
    # =========================
    if countdown > 0 and time.time() - countdown_start >= 1:
        countdown -= 1
        countdown_start = time.time()

        if countdown == 0:
            locked_player_move = player_move
            pc_move = random.choice(CLASSES)
            result = decide_winner(locked_player_move, pc_move)

            if result == "HAI VINTO":
                player_score += 1
            elif result == "HAI PERSO":
                pc_score += 1

            pred_buffer.clear()
            reset_timer = time.time()  # parte il timer per resettare dopo 1 secondo

    # =========================
    # RESET AUTOMATICO DOPO 1 SECONDO
    # =========================
    if reset_timer is not None and time.time() - reset_timer >= 2:
        locked_player_move = None
        pc_move = None
        player_move = "?"
        confidence = 0.0
        reset_timer = None

    # =========================
    # COMPOSIZIONE SCENA e HUD
    # =========================
    frame = cv2.resize(bg_img, (w, h))

    SCREEN_X = (w - SCREEN_W) // 2 + SCREEN_X_OFF
    cam_screen = cv2.resize(roi, (SCREEN_W, SCREEN_H))
    sx1 = max(0, SCREEN_X)
    sy1 = max(0, SCREEN_Y)
    sx2 = min(w, SCREEN_X + SCREEN_W)
    sy2 = min(h, SCREEN_Y + SCREEN_H)
    cam_crop = cam_screen[0:(sy2 - sy1), 0:(sx2 - sx1)]
    frame[sy1:sy2, sx1:sx2] = cam_crop

    HUD_X, HUD_Y = 40, 50
    # MOSTRA SEMPRE "Tu: ?" FINCHÉ IL COUNTDOWN NON È FINITO
    if locked_player_move is not None and countdown == 0:
        draw_text(frame, f"Tu: {locked_player_move}", (HUD_X, HUD_Y), 1.0, (0,255,0), 2)
    else:
        draw_text(frame, "Tu: ?", (HUD_X, HUD_Y), 1.0, (0,255,0), 2)


    if countdown > 0:
        bar_color = (0,200,0) if confidence >= 0.75 else (0,200,200) if confidence >= 0.5 else (0,0,200)
        bar_w, bar_h = 200, 10
        bar_y = HUD_Y + 25
        cv2.rectangle(frame, (HUD_X, bar_y), (HUD_X + bar_w, bar_y + bar_h), (70,70,70), -1)
        cv2.rectangle(frame, (HUD_X, bar_y), (HUD_X + int(bar_w * confidence), bar_y + bar_h), bar_color, -1)
        draw_text(frame, f"{int(confidence*100)}%", (HUD_X + bar_w + 10, bar_y + 10), 0.55, (220,220,220), 1)

    if pc_move:
        draw_text(frame, f"AI: {pc_move}", (HUD_X, HUD_Y + 70), 1.0, (255,80,80), 2)
        draw_text(frame, result, (HUD_X, HUD_Y + 105), 1.2, (255,220,120), 2)

    if countdown > 0:
        draw_text(frame, str(countdown), (SCREEN_X + SCREEN_W//2 - 20, SCREEN_Y + SCREEN_H//2 + 15), 3, (255,0,0), 4)

    draw_text(frame, f"{player_score} : {pc_score}", (SCREEN_X + SCREEN_W//2 - 40, SCREEN_Y - 20), 1.2, (255,255,255), 2)
    draw_text(frame, "SPAZIO = Gioca   Q = Esci", (SCREEN_X + 40, h - 30), 0.6, (220,220,220), 1)

    cv2.imshow(WINDOW_NAME, frame)

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
