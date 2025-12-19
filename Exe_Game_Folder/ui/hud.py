import cv2
from ui.draw import draw_text

def draw_hp(frame, hp, max_hp, x, y, color):
    w = int(200 * hp / max_hp)
    cv2.rectangle(frame,(x,y),(x+200,y+16),(60,60,60),-1)
    cv2.rectangle(frame,(x,y),(x+w,y+16),color,-1)

def draw_combo(frame, combo, x, y, text):
    if combo>=2:
        draw_text(frame,text,(x,y),0.7,(255,215,0),2)
