import numpy as np
import time

def screen_shake(strength=6):
    return np.random.randint(-strength,strength+1), np.random.randint(-strength,strength+1)

def flash(frame, color, alpha=0.18):
    overlay = frame.copy()
    overlay[:] = color
    return (overlay*alpha + frame*(1-alpha)).astype("uint8")
