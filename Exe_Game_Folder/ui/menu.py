import cv2

MENU_ITEMS = ["START", "CREDITI", "ESCI"]

def run_menu(window_name, bg):
    idx = 0

    while True:
        frame = bg.copy()

        cv2.putText(frame, "ROCK PAPER SCISSORS",
                    (360,120), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (255,255,255), 3)

        for i, item in enumerate(MENU_ITEMS):
            color = (0,255,0) if i == idx else (200,200,200)
            cv2.putText(frame, item,
                        (560, 260 + i*60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, color, 3)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 82:   # ↑
            idx = (idx - 1) % len(MENU_ITEMS)
        elif key == 84: # ↓
            idx = (idx + 1) % len(MENU_ITEMS)
        elif key == ord(" "):
            return MENU_ITEMS[idx]
