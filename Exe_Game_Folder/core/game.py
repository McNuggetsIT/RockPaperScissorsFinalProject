import cv2, torch, random, time
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from collections import deque
import numpy as np
import os

from systems.audio import *
from systems.combat import *
from systems.effects import *
from ui.draw import draw_text
from ui.hud import draw_hp, draw_combo


# =========================
# PATH & COSTANTI
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "assets", "GAG_RPS_Model.pth")
BG_PATH    = os.path.join(BASE_DIR, "assets", "Bg23.png")

CLASSES = ["rock", "paper", "scissors"]
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# MODELLO AI
# =========================
class RPSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*30*30,128), nn.ReLU(),
            nn.Linear(128,3)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# LOGICA RPS
# =========================
def decide(p, a):
    if p == a:
        return "DRAW"
    if (p=="rock" and a=="scissors") or \
       (p=="paper" and a=="rock") or \
       (p=="scissors" and a=="paper"):
        return "PLAYER"
    return "AI"


# =========================
# GAME LOOP
# =========================
def run_game():

    # ----- MODELLO -----
    model = RPSNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # ----- ASSETS -----
    bg = cv2.imread(BG_PATH)
    if bg is None:
        raise FileNotFoundError(BG_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera non disponibile")

    cap.set(3,1280)
    cap.set(4,720)

    # ----- STATO GIOCO -----
    player_hp = ai_hp = MAX_HP
    p_combo = a_combo = 0
    p_score = a_score = 0

    roi_w = roi_h = 340
    screen_w, screen_h = 425, 230
    screen_y = 220
    screen_x_off = 213

    pred_buffer = deque(maxlen=5)

    countdown = 0
    cd_time = 0
    locked = None

    shake_t = 0
    flash_t = 0
    flash_color = None

    last_result = None
    result_time = 0

    player_move = "rock"

    # ----- TRANSFORM -----
    tf = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # =========================
    # LOOP PRINCIPALE
    # =========================
    while True:
        ret, cam = cap.read()
        if not ret:
            break

        cam = cv2.flip(cam, 1)
        h, w, _ = cam.shape

        # ----- ROI -----
        cy, cx = h//2, w//2
        roi = cam[
            cy-roi_h//2 : cy+roi_h//2,
            cx-roi_w//2 : cx+roi_w//2
        ]
        roi = cv2.resize(roi, (roi_w, roi_h))

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        img = tf(
            Image.fromarray(
                cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
            )
        ).unsqueeze(0).to(DEVICE)

        # ----- PREDIZIONE -----
        if countdown == 0:
            with torch.no_grad():
                p = F.softmax(model(img), 1)[0]
                pred_buffer.append(CLASSES[p.argmax().item()])
                player_move = max(set(pred_buffer), key=pred_buffer.count)

        # ----- INPUT -----
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            snd_click.play()
            countdown = 3
            cd_time = time.time()
            locked = player_move
            pred_buffer.clear()

        if key == ord("q"):
            break

        # ----- COUNTDOWN & COMBAT -----
        if countdown > 0 and time.time() - cd_time >= 1:
            snd_count.play()
            countdown -= 1
            cd_time = time.time()

            if countdown == 0:
                ai_move = random.choice(CLASSES)
                win = decide(locked, ai_move)

                last_result = win
                result_time = time.time()

                if win == "PLAYER":
                    p_combo += 1
                    a_combo = 0
                    ai_hp = max(0, ai_hp - calc_damage(p_combo))
                    snd_hit_ai.play()
                    shake_t = flash_t = time.time()
                    flash_color = (0,255,0)

                elif win == "AI":
                    a_combo += 1
                    p_combo = 0
                    player_hp = max(0, player_hp - calc_damage(a_combo))
                    snd_hit_player.play()
                    shake_t = flash_t = time.time()
                    flash_color = (0,0,255)

                else:
                    p_combo = a_combo = 0

                if ai_hp == 0:
                    p_score += 1
                    snd_win.play()
                    ai_hp = player_hp = MAX_HP

                if player_hp == 0:
                    a_score += 1
                    snd_lose.play()
                    ai_hp = player_hp = MAX_HP

        # =========================
        # RENDERING
        # =========================
        frame = cv2.resize(bg, (w, h))

        dx = dy = 0
        if time.time() - shake_t < 0.25:
            dx, dy = screen_shake()

        if time.time() - flash_t < 0.15 and flash_color:
            frame = flash(frame, flash_color)

        sx = (w - screen_w)//2 + screen_x_off + dx
        sy = screen_y + dy
        frame[sy:sy+screen_h, sx:sx+screen_w] = cv2.resize(
            roi, (screen_w, screen_h)
        )

        # ----- HUD -----
        draw_text(frame, f"Tu: {player_move}", (40,45), 1, (0,255,0), 2)

        draw_text(frame, "HP", (40,120), 0.8, (255,255,255), 2)
        draw_text(frame, "AI HP", (w-240,120), 0.8, (255,255,255), 2)

        draw_hp(frame, player_hp, MAX_HP, 40, 140, (0,200,0))
        draw_hp(frame, ai_hp, MAX_HP, w-240, 140, (0,0,200))

        draw_combo(frame, p_combo, 40, 180, "COMBO x2!")
        draw_combo(frame, a_combo, w-240, 180, "AI COMBO x2!")

        draw_text(
            frame,
            f"{p_score} : {a_score}",
            (sx + screen_w//2 - 40, sy - 20),
            1.2
        )

        # ----- COUNTDOWN VISIVO (IN ALTO) -----
        if countdown > 0:
            draw_text(
                frame,
                str(countdown),
                (w//2 - 30, 90),
                2.5,
                (255,255,255),
                4
            )


        # ----- HIT / MISS -----
        if time.time() - result_time < 0.6:
            if last_result == "PLAYER":
                draw_text(frame, "HIT!", (w//2-60, 100), 1.2, (0,255,0), 3)
            elif last_result == "AI":
                draw_text(frame, "MISS!", (w//2-80, 100), 1.2, (0,0,255), 3)

        cv2.imshow("Rock Paper Scissors", frame)

    cap.release()
    cv2.destroyAllWindows()
