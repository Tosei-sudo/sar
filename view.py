# coding: utf-8
import time

import hashlib
import numpy as np
import matplotlib.pyplot as plt

# for 3D plot
from mpl_toolkits.mplot3d import Axes3D

# for 3d surface plot
from matplotlib import cm

from model.trigonometric import Trigonometric
from model.surface import Surface

surfaces = [
    [
        [0, 0, 0],
        [0, 0, 5],
        [5, 0, 5],
        [5, 0, 0],
        [0, 0, 0],
    ] ,
    [
        [0, 0, 0],
        [0, 0, 5],
        [0, 10, 5],
        [0, 10, 0],
        [0, 0, 0],
    ] ,
    [
        [5, 0, 0],
        [5, 0, 5],
        [5, 10, 5],
        [5, 10, 0],
        [5, 0, 0],
    ] ,
    [
        [0, 0, 5],
        [2.5, 1, 7],
        [5, 0, 5],
        [0, 0, 5],
    ] ,  
    [
        [0, 10, 5],
        [2.5, 9, 7],
        [5, 10, 5],
        [0, 10, 5],
    ] ,  
    [
        [2.5, 1, 7],
        [2.5, 9, 7],
        [5, 10, 5],
        [5, 0, 5],
        [2.5, 1, 7],
    ],
    [
        [2.5, 1, 7],
        [2.5, 9, 7],
        [0, 10, 5],
        [0, 0, 5],
        [2.5, 1, 7],
    ] ,
    [
        [0, 10, 0],
        [0, 10, 5],
        [5, 10, 5],
        [5, 10, 0],
        [0, 10, 0],
    ] ,
    [
        [2, 12, 0],
        [2, 12, 5],
        [4, 12, 5],
        [4, 12, 0],
        [2, 12, 0],
    ] ,
    [
        [2, 12, 0],
        [2, 12, 5],
        [2, 14, 5],
        [2, 14, 0],
        [2, 12, 0],
    ] ,
    [
        [2, 14, 0],
        [2, 14, 5],
        [4, 14, 5],
        [4, 14, 0],
        [2, 14, 0],
    ] ,
    [
        [4, 14, 0],
        [4, 14, 5],
        [4, 12, 5],
        [4, 12, 0],
        [4, 14, 0],
    ] ,
    [
        [12, 12, 0],
        [12, 12, 5],
        [14, 12, 5],
        [14, 12, 0],
        [12, 12, 0],
    ] ,
    [
        [12, 12, 0],
        [12, 12, 5],
        [12, 14, 5],
        [12, 14, 0],
        [12, 12, 0],
    ] ,
    [
        [12, 14, 0],
        [12, 14, 5],
        [14, 14, 5],
        [14, 14, 0],
        [12, 14, 0],
    ] ,
    [
        [14, 14, 0],
        [14, 14, 5],
        [14, 12, 5],
        [14, 12, 0],
        [14, 14, 0],
    ] ,
    [
        [12, 12, 5],
        [13, 13, 10],
        [14, 12, 5],
        [12, 12, 5],
    ],
    [
        [14, 12, 5],
        [13, 13, 10],
        [14, 14, 5],
        [14, 12, 5],
    ],
    [
        [14, 14, 5],
        [13, 13, 10],
        [12, 14, 5],
        [14, 14, 5],
    ],
    [
        [12, 14, 5],
        [13, 13, 10],
        [12, 12, 5],
        [12, 14, 5],
    ],
]

ga = 40
shadow_angle = 70
layover_angle = 255

shadow_model = Trigonometric(shadow_angle)
ga_model = Trigonometric(ga)
layover_model = Trigonometric(layover_angle)

beam_span = 0.15

def plot_3d_surface():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    beam_lines = []
    surface_beams_map = {}

    for surface in surfaces:
        surface_key = hashlib.sha256(str(surface)).hexdigest()
        
        surface = np.array(surface, dtype=np.float)
        
        surface_beams = []
        for i in range(len(surface) - 1):
            
            if np.array_equal(surface[i][:2], surface[i + 1][:2]):
                continue
            
            surface_beams.append([
                surface[i],
                surface[i + 1],
            ])
        
        beam_lines.extend(surface_beams)
        
        surface_beams_map[surface_key] = {(tuple(key[0]), tuple(key[1])): True for key in surface_beams}

    beam_lines = np.array(beam_lines)
    
    for surface in surfaces:
        surface_key = hashlib.sha256(str(surface)).hexdigest()
        
        surface = np.array(surface, dtype=np.float)
        s = Surface(surface)
        
        ax.plot(surface[:, 0], surface[:, 1], surface[:, 2], 'r')
        
        shadow_points = np.copy(surface[:-1])
        shadow_points[:, 0] += shadow_model.cos * shadow_points[:, 2] / ga_model.tan
        shadow_points[:, 1] += shadow_model.sin * shadow_points[:, 2] / ga_model.tan
        shadow_points[:, 2] = 0
        ax.plot(shadow_points[:, 0], shadow_points[:, 1], shadow_points[:, 2], 'gray')
        
        layuover_points = np.copy(surface)
        layuover_points[:, 0] += layover_model.cos * layuover_points[:, 2] / ga_model.tan
        layuover_points[:, 1] += layover_model.sin * layuover_points[:, 2] / ga_model.tan
        layuover_points[:, 2] = 0
        ax.plot(layuover_points[:, 0], layuover_points[:, 1], layuover_points[:, 2], 'g')
    
        surface_beams = surface_beams_map.get(surface_key, [])
        
        for beam_line in beam_lines:
            beam_key = (tuple(beam_line[0]), tuple(beam_line[1]))
            if beam_key in surface_beams:
                continue
            
            length = np.linalg.norm(beam_line[1] - beam_line[0])
            
            splited_length = np.tile(np.arange(0, length, beam_span), (3, 1)).T
            
            beam_start_points = np.zeros((len(splited_length), 3))
            beam_start_points[:] = (beam_line[1] - beam_line[0]) * (splited_length / length)
            
            beam_start_points += beam_line[0]
            
            shadow_lengths = beam_start_points[:, 2] / ga_model.tan
            
            beam_end_points = np.copy(beam_start_points)
            beam_end_points[:, 0] += shadow_model.cos * shadow_lengths[:]
            beam_end_points[:, 1] += shadow_model.sin * shadow_lengths[:]
            beam_end_points[:, 2] = 0
            
            intersection_points, _ = s.get_intersection_points(beam_start_points, beam_end_points, beam_span)

            if intersection_points is None:
                continue
            
            # ax.scatter(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2], marker='o', color='blue')
    
    plt.show()

if __name__ == '__main__':
    start = time.time()
    
    plot_3d_surface()
    
    print('elapsed time: ', time.time() - start)