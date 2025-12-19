import cv2
import time
import numpy as np

def show_splash(window_name, splash_path, logo_path, size=(1280,720)):
    splash = cv2.imread(splash_path)
    logo = cv2.imread(logo_path)

    splash = cv2.resize(splash, size)
    logo = cv2.resize(logo, size)

    # mostra splash
    cv2.imshow(window_name, splash)
    cv2.waitKey(1)
    time.sleep(2.0)

    # dissolve splash → logo
    for a in np.linspace(0, 1, 30):
        frame = cv2.addWeighted(splash, 1-a, logo, a, 0)
        cv2.imshow(window_name, frame)
        cv2.waitKey(30)

    # logo fermo
    cv2.imshow(window_name, logo)
    cv2.waitKey(1)
    time.sleep(1.5)
