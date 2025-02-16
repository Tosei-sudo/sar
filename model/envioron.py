# coding: utf-8
import numpy as np

from .trigonometric import Trigonometric

class Envioron:
    def __init__(self, grazing_angle, shadow_angle, layover_angle, x_span=0.06, y_span=0.06, canvas_span=0.12):
        self.grazing_angle = grazing_angle
        self.shadow_angle = shadow_angle
        self.layover_angle = layover_angle
        
        self.grazing = Trigonometric(self.grazing_angle)
        self.shadow = Trigonometric(self.shadow_angle)
        self.layover = Trigonometric(self.layover_angle)
        
        self.x_span = x_span
        self.y_span = y_span
        
        self.canvas_span = canvas_span
        
        self.beam_span = np.linalg.norm([x_span, y_span])