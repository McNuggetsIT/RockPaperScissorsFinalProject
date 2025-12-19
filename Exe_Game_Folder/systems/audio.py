import pygame
import numpy as np

pygame.mixer.init(44100, -16, 2)

def gen_tone(freq, dur, vol=0.4):
    t = np.linspace(0, dur, int(44100*dur), False)
    wave = np.sin(freq * t * 2*np.pi)
    wave = (wave*(2**15-1)*vol).astype(np.int16)
    stereo = np.column_stack((wave,wave))
    return pygame.sndarray.make_sound(stereo)

snd_hit_ai     = gen_tone(220,0.12)
snd_hit_player = gen_tone(90,0.15)
snd_win        = gen_tone(880,0.35)
snd_lose       = gen_tone(140,0.4)
snd_count      = gen_tone(600,0.08)
snd_click      = gen_tone(500,0.05)
