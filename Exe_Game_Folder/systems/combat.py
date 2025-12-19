MAX_HP = 100
BASE_DMG = 10

def calc_damage(combo):
    return BASE_DMG * 2 if combo >= 2 else BASE_DMG
