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
import numpy as np
import pygame
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# =========================
# AUDIO AUTO-GENERATO (FIX STEREO)
# =========================
import pygame
import numpy as np

pygame.mixer.init(frequency=44100, size=-16, channels=2)

def gen_tone(freq=440, duration=0.15, volume=0.4):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    wave = np.sin(freq * t * 2 * np.pi)
    wave = (wave * (2**15 - 1) * volume).astype(np.int16)

    # 🔥 DUPLICA IN STEREO (shape: [N, 2])
    stereo_wave = np.column_stack((wave, wave))

    return pygame.sndarray.make_sound(stereo_wave)

# =========================
# SUONI DI GIOCO
# =========================
snd_hit_ai     = gen_tone(220, 0.12)   # colpisci AI
snd_hit_player = gen_tone(90, 0.15)    # vieni colpito
snd_win        = gen_tone(880, 0.35)   # vittoria
snd_lose       = gen_tone(140, 0.40)   # sconfitta
snd_count      = gen_tone(600, 0.08)   # countdown
snd_click      = gen_tone(500, 0.05)   # click


# =========================
# PARAMETRI
# =========================
IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["rock", "paper", "scissors"]

MAX_HP = 100
DMG = 25
HP_LERP_SPEED = 0.15

# =========================
# PATH
# =========================
MODEL_PATH = os.path.join(BASE_DIR,"Working_Game_Folder","GAG_RPS_Model.pth")
BG_PATH    = os.path.join(BASE_DIR,"Working_Game_Folder","Bg23.png")

# =========================
# MODELLO
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
    def forward(self,x): 
        return self.net(x)

def decide_round(p,a):
    if p == a: 
        return "DRAW"
    if (p=="rock" and a=="scissors") or (p=="paper" and a=="rock") or (p=="scissors" and a=="paper"):
        return "PLAYER"
    return "AI"

def draw_text(img,text,pos,scale=1,color=(255,255,255),th=2):
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),th+2)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,color,th)

# =========================
# LOAD
# =========================
model = RPSNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
model.eval()

bg = cv2.imread(BG_PATH)

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# =========================
# STATO
# =========================
player_hp = ai_hp = MAX_HP
player_hp_vis = ai_hp_vis = MAX_HP

player_score = ai_score = 0
player_combo = ai_combo = 0

player_move = "?"
ai_move = None
confidence = 0

countdown = 0
countdown_start = 0
locked_move = None
round_done = False

game_result = ""
result_time = 0

pred_buffer = deque(maxlen=5)
shake_time = 0
flash_color = None

# =========================
# ROI / SCREEN
# =========================
ROI_W, ROI_H = 340, 340
SCREEN_W, SCREEN_H = 425, 230
SCREEN_Y = 220
SCREEN_X_OFF = 213

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

def calc_damage(combo):
    return DMG * 2 if combo >= 2 else DMG

# =========================
# LOOP
# =========================
while True:
    ret, cam = cap.read()
    if not ret:
        break

    cam = cv2.flip(cam,1)
    h,w,_ = cam.shape

    cy,cx = h//2, w//2
    roi = cam[cy-ROI_H//2:cy+ROI_H//2, cx-ROI_W//2:cx+ROI_W//2]
    roi = cv2.resize(roi,(ROI_W,ROI_H))

    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    _,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    img = transform(Image.fromarray(cv2.cvtColor(th,cv2.COLOR_GRAY2RGB))).unsqueeze(0).to(DEVICE)

    if countdown==0 and game_result=="":
        with torch.no_grad():
            probs = F.softmax(model(img),1)[0]
            pred = torch.argmax(probs).item()
            confidence = probs[pred].item()
            pred_buffer.append(CLASSES[pred])
            player_move = max(set(pred_buffer),key=pred_buffer.count)

    key = cv2.waitKey(1)&0xFF
    if key==ord(" ") and countdown==0 and game_result=="":
        snd_click.play()
        countdown=3
        countdown_start=time.time()
        locked_move=player_move
        pred_buffer.clear()
        round_done=False

    if key==ord("q"):
        break

    if countdown>0 and time.time()-countdown_start>=1:
        countdown-=1
        countdown_start=time.time()
        snd_count.play()

        if countdown==0 and not round_done:
            ai_move=random.choice(CLASSES)
            win=decide_round(locked_move,ai_move)

            if win=="PLAYER":
                player_combo += 1
                ai_combo = 0
                ai_hp = max(0, ai_hp - calc_damage(player_combo))
                snd_hit_ai.play()
                flash_color=(0,255,0)
                shake_time=time.time()

            elif win=="AI":
                ai_combo += 1
                player_combo = 0
                player_hp = max(0, player_hp - calc_damage(ai_combo))
                snd_hit_player.play()
                flash_color=(0,0,255)
                shake_time=time.time()

            else:
                player_combo = 0
                ai_combo = 0

            round_done=True

            if ai_hp==0:
                player_score+=1
                snd_win.play()
                game_result="HAI VINTO"
                result_time=time.time()

            elif player_hp==0:
                ai_score+=1
                snd_lose.play()
                game_result="HAI PERSO"
                result_time=time.time()

    if game_result!="" and time.time()-result_time>2.5:
        player_hp=ai_hp=MAX_HP
        player_hp_vis=ai_hp_vis=MAX_HP
        player_combo=ai_combo=0
        game_result=""
        ai_move=None

    # =========================
    # HP SMOOTH
    # =========================
    player_hp_vis += (player_hp-player_hp_vis)*HP_LERP_SPEED
    ai_hp_vis += (ai_hp-ai_hp_vis)*HP_LERP_SPEED

    # =========================
    # RENDER
    # =========================
    frame=cv2.resize(bg,(w,h))

    dx=dy=0
    if time.time()-shake_time<0.25:
        dx=np.random.randint(-6,7)
        dy=np.random.randint(-6,7)

    sx=(w-SCREEN_W)//2+SCREEN_X_OFF+dx
    sy=SCREEN_Y+dy
    frame[sy:sy+SCREEN_H, sx:sx+SCREEN_W]=cv2.resize(roi,(SCREEN_W,SCREEN_H))

    # HUD
    draw_text(frame,f"Tu: {player_move}",(40,45),1,(0,255,0),2)

    cv2.rectangle(frame,(40,65),(240,75),(70,70,70),-1)
    cv2.rectangle(frame,(40,65),(40+int(200*confidence),75),(0,200,0),-1)

    if ai_move:
        draw_text(frame,f"AI: {ai_move}",(40,105),1,(255,80,80),2)

    # PLAYER HP
    draw_text(frame,"HP",(40,135),0.6)
    cv2.rectangle(frame,(40,140),(240,156),(60,60,60),-1)
    cv2.rectangle(frame,(40,140),(40+int(200*player_hp_vis/MAX_HP),156),(0,200,0),-1)

    # AI HP (ROSSA)
    draw_text(frame,"AI HP",(w-240,135),0.6)
    cv2.rectangle(frame,(w-240,140),(w-40,156),(60,60,60),-1)
    cv2.rectangle(frame,(w-240,140),(w-240+int(200*ai_hp_vis/MAX_HP),156),(0,0,200),-1)

    # COMBO HUD
    if player_combo>=2:
        draw_text(frame,"COMBO x2!",(40,180),0.7,(255,215,0),2)
    if ai_combo>=2:
        draw_text(frame,"AI COMBO x2!",(w-240,180),0.7,(255,80,80),2)

    draw_text(frame,f"{player_score} : {ai_score}",(sx+SCREEN_W//2-40,sy-20),1.2)

    if countdown>0:
        draw_text(frame,str(countdown),(sx+SCREEN_W//2-20,sy+SCREEN_H//2),3,(255,0,0),4)

    if game_result!="":
        draw_text(frame,game_result,(sx+80,sy+SCREEN_H//2),1.6,(255,220,120),3)

    draw_text(frame,"SPAZIO = Gioca   Q = Esci",(sx+40,h-30),0.6)

    if flash_color and time.time()-shake_time<0.15:
        overlay=frame.copy()
        overlay[:]=flash_color
        cv2.addWeighted(overlay,0.18,frame,0.82,0,frame)

    cv2.imshow("Rock Paper Scissors",frame)

cap.release()
cv2.destroyAllWindows()
