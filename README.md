# 🎮 GAG – Rock Paper Scissors: AI Edition

GAG – Rock Paper Scissors: AI Edition è un gioco interattivo che unisce Computer Vision, Intelligenza Artificiale e Game Design in un’esperienza ispirata a Rock Paper Scissors, con una narrativa distopica in cui l’umanità affronta una super-AI chiamata DOT.
Il giocatore usa i gesti della mano tramite webcam per sfidare l’AI in tempo reale.

---

## 🧠 Concept

In un futuro dominato dall’intelligenza artificiale, DOT controlla le infrastrutture del mondo e minaccia l’umanità.  
Il giocatore è l’ultimo umano in grado di affrontarla.

Ogni partita è una battaglia tra:
- Umano → gesto reale
- AI → rete neurale

---

## 🛠️ Tecnologie

- Python 3.10  
- OpenCV  
- PyTorch (CNN)  
- NumPy  
- PIL  
- PyInstaller (build .exe)

---

## 🤖 AI – Gesture Recognition

Il modello GAG_RPS_Model.pth è una CNN addestrata per riconoscere:
- Rock  
- Paper  
- Scissors  

Pipeline:
1. Webcam → ROI centrale  
2. Grayscale  
3. Thresholding  
4. Resize a 128×128  
5. Normalizzazione  
6. Predizione CNN  
7. Media su più frame per stabilità  

---

## 🎮 Gameplay

- HP Player vs HP AI  
- Sistema di combo  
- Danni crescenti  
- Effetti visivi (flash, shake, HIT / MISS)  
- Countdown prima dello scontro  

---

## 🗂️ Struttura del progetto

Exe_Game_Folder  
├── assets  
│   ├── GAG_RPS_Model.pth  
│   ├── Bg23.png  
│   ├── splash.png  
│   └── logo.png  
├── core  
│   ├── game.py  
│   └── menu.py  
├── systems  
│   ├── audio.py  
│   ├── combat.py  
│   └── effects.py  
├── ui  
│   ├── draw.py  
│   ├── hud.py  
│   ├── splash.py  
│   └── menu.py  
└── main.py  

---

## ▶️ Come eseguire

### Metodo 1 — Python
pip install opencv-python torch torchvision numpy pillow pygame
python main.py  

Assicurati che la webcam sia collegata e che il file GAG_RPS_Model.pth sia nella cartella assets.

### Metodo 2 — .EXE
Avvia direttamente il file eseguibile generato con PyInstaller.

---


## 🚀 Obiettivo

Questo progetto esplora:
- interazione uomo-macchina  
- gesture control  
- AI applicata al gameplay  

Unendo programmazione, intelligenza artificiale e storytelling.

---

## 📌 Autore

Sviluppato come progetto durante il percorso formativo di IT Consulting.
