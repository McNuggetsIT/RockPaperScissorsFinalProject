import cv2

def draw_text(img,text,pos,scale=1,color=(255,255,255),th=2):
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,
                scale,(0,0,0),th+2)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,
                scale,color,th)
