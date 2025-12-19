import cv2
import os

from ui.splash import show_splash
from core.menu import run_menu
from core.game import run_game

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(BASE_DIR, "assets")

SPLASH_PATH = os.path.join(ASSETS, "splash.png")
LOGO_PATH   = os.path.join(ASSETS, "logo.png")

WINDOW = "Rock Paper Scissors"

def main():
    # 1 crea la finestra UNA SOLA VOLTA
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    # SPLASH + TRANSIZIONE LOGO (SOLO ALL'AVVIO)
    show_splash(WINDOW, SPLASH_PATH, LOGO_PATH)

    #  MENU
    while True:
        choice = run_menu()

        if choice == "START":
            run_game()

        elif choice == "EXIT":
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

