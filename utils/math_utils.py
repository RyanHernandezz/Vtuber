from math import hypot

def euclid(p1, p2):
    return hypot(p1[0]-p2[0], p1[1]-p2[1])

def clamp01(x):
    return max(0.0, min(1.0, x))

