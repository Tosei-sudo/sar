# coding: utf-8
import math

class Trigonometric:
    def __init__(self, angle):
        self.angle = angle
        self.rad = math.radians(angle)
        
        self.sin = math.sin(self.rad)
        self.cos = math.cos(self.rad)
        self.tan = math.tan(self.rad)