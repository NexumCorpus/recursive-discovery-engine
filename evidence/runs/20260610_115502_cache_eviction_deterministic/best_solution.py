def _evict_score(la, fr, ins, t):
    return la + 30 * fr.bit_length()

def _on_hit(la, fr, ins, t):
    return [t, fr + 2, ins]
CMUL = 4
