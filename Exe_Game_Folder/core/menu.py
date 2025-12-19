import cv2
import numpy as np
from ui.draw import draw_text

WINDOW = "Rock Paper Scissors"

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
    cv2.namedWindow(WINDOW)

    while True:
        frame = np.zeros((720,1280,3),dtype=np.uint8)

        draw_text(frame,"GAG - ROCK PAPER SCISSORS",(320,150),1.6,(255,215,0),3)

        for i,item in enumerate(menu_items):
            color = (0,255,0) if i==idx else (200,200,200)
            draw_text(frame,item,(520,260+i*60),1.2,color,2)

        cv2.imshow(WINDOW,frame)
        key = cv2.waitKey(1)&0xFF

        if key==ord("w"): idx=(idx-1)%len(menu_items)
        if key==ord("s"): idx=(idx+1)%len(menu_items)
        if key==13:  # ENTER
            if menu_items[idx]=="START":
                return "START"
            if menu_items[idx]=="CREDITI":
                show_credits()
            if menu_items[idx]=="ESCI":
                return "EXIT"

def show_credits():
    while True:
        frame = np.zeros((720,1280,3),dtype=np.uint8)
        y = 140
        for line in credits:
            draw_text(frame,line,(400,y),0.9,(220,220,220),2)
            y+=40

        draw_text(frame,"PREMI Q PER TORNARE",(400,600),0.8,(255,255,255),2)
        cv2.imshow(WINDOW,frame)

        if cv2.waitKey(1)&0xFF==ord("q"):
            break
