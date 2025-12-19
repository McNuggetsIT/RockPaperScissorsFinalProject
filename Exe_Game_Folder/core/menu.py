import cv2
import os
import numpy as np
from ui.draw import draw_text

WINDOW = "Rock Paper Scissors"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS   = os.path.join(BASE_DIR, "assets")

MENU_BG_PATH = os.path.join(ASSETS, "menu.png")

menu_items = ["START", "CREDITI", "ESCI"]

credits = [
    "GAG - Rock Paper Scissors",
    "",
    "Game Design & Code:",
    " - TU",
    "",
    "AI:",
    " - PyTorch CNN",
    "",
    "Audio:",
    " - Procedural FX",
    "",
    "Powered by Python"
]


def run_menu():
    idx = 0

    # ✅ CARICA UNA SOLA VOLTA
    menu_bg = cv2.imread(MENU_BG_PATH)
    if menu_bg is None:
        raise FileNotFoundError(f"Menu background non trovato: {MENU_BG_PATH}")

    while True:
        # ✅ USA menu.png
        frame = cv2.resize(menu_bg, (1280, 720))

        draw_text(
            frame,
            "GAG - ROCK PAPER SCISSORS",
            (300, 140),
            1.6,
            (255, 215, 0),
            3
        )

        for i, item in enumerate(menu_items):
            color = (0, 255, 0) if i == idx else (200, 200, 200)
            draw_text(
                frame,
                item,
                (520, 260 + i * 60),
                1.2,
                color,
                2
            )

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("w"):
            idx = (idx - 1) % len(menu_items)

        if key == ord("s"):
            idx = (idx + 1) % len(menu_items)

        if key == 13:  # ENTER
            if menu_items[idx] == "START":
                return "START"

            if menu_items[idx] == "CREDITI":
                show_credits(menu_bg)

            if menu_items[idx] == "ESCI":
                return "EXIT"


def show_credits(menu_bg):
    while True:
        frame = cv2.resize(menu_bg, (1280, 720))

        y = 180
        for line in credits:
            draw_text(frame, line, (420, y), 0.9, (220, 220, 220), 2)
            y += 40

        draw_text(frame, "PREMI Q PER TORNARE", (420, 620), 0.8, (255, 255, 255), 2)

        cv2.imshow(WINDOW, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
