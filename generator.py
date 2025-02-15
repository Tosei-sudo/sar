# coding: utf-8
import math
import time
import numpy as np

from model.trigonometric import Trigonometric
from model.surface import Surface

import matplotlib.pyplot as plt
from scipy.ndimage import zoom

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
        [2, 12, 5],
        [2, 14, 5],
        [4, 14, 5],
        [4, 12, 5],
        [2, 12, 5],
    ],
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
        [12, 14, 5],
        [14, 14, 5],
        [14, 12, 5],
        [12, 12, 5],
    ]
]
start = time.time()

ga = 45
shadow_angle = 90
layover_angle = 225

x_span = 0.03
y_span = 0.03

canvas_span = 0.06

shadow_model = Trigonometric(shadow_angle)
ga_model = Trigonometric(ga)
layover_model = Trigonometric(layover_angle)

x_range = [np.inf, -np.inf]
y_range = [np.inf, -np.inf]
z_range = [np.inf, -np.inf]

for surface in surfaces:
    surface = np.array(surface)
    
    x_range = [min(x_range[0], surface[:, 0].min()), max(x_range[1], surface[:, 0].max())]
    y_range = [min(y_range[0], surface[:, 1].min()), max(y_range[1], surface[:, 1].max())]
    z_range = [min(z_range[0], surface[:, 2].min()), max(z_range[1], surface[:, 2].max())]

beam_span = np.linalg.norm([x_span, y_span])

buffer_norm = int(z_range[1] * ga_model.tan * 2) + 1
buffer = int(max(
    buffer_norm * shadow_model.cos,
    buffer_norm * shadow_model.sin,
    buffer_norm * layover_model.cos,
    buffer_norm * layover_model.sin
))

x_list = np.arange(x_range[0] - buffer, x_range[1] + buffer, x_span)
y_list = np.arange(y_range[0] - buffer, y_range[1] + buffer, y_span)

end_points = np.array([[x, y, 0] for x in x_list for y in y_list])

# 簡易計算の実装　ビームの視点を一定の距離に設定
s_range = z_range[1] * ga_model.sin * 2

start_points = np.copy(end_points)
start_points[:, 0] -= shadow_model.cos * s_range
start_points[:, 1] -= shadow_model.sin * s_range
start_points[:, 2] += ga_model.tan * s_range

# 精密計算の実装　ビームの視点を一点に設定
# s_range = 2e2

# start_points = np.zeros_like(end_points)
# start_points[:, 0] = np.mean(x_range) - shadow_model.cos * s_range
# start_points[:, 1] = np.mean(y_range) - shadow_model.sin * s_range
# start_points[:, 2] = ga_model.tan * s_range

# 実装メモ
'''
サーフェイスごとにビームの衝突を判定
交点を記録し、始点に一番近い交点を反射点に選択
反射点をレイオーバーを考慮して2次元に投影する
'''

distance_map = np.zeros((len(start_points), len(surfaces)))
distance_map.fill(np.inf)

intersection_map = np.zeros((len(start_points), len(surfaces), 3))

for (index, surface) in enumerate(surfaces):
    sur = Surface(np.array(surface, dtype=np.float))
    
    points, indexs = sur.get_intersection_points(start_points, end_points, beam_span)
    
    if points is None:
        continue
    distance = np.repeat(np.inf, len(start_points))
    
    d = np.linalg.norm(points - start_points[indexs], axis=1)
    distance[indexs] = d

    distance_map[:, index] = distance

    intersection_map[indexs, index] = points

# 条件を満たすインデックスを取得
inf_mask = np.all(distance_map == np.inf, axis=1)

# `groud_layover_points` の初期化
groud_layover_points = np.zeros((len(start_points), 2))

# 条件を満たす部分を直接代入
groud_layover_points[inf_mask] = end_points[inf_mask, :2]

# 条件を満たさない部分の処理
valid_indices = ~inf_mask
near_point_indices = np.argmin(distance_map[valid_indices], axis=1)
near_points = intersection_map[valid_indices, near_point_indices]

groud_layover_points[valid_indices] = np.stack([
    near_points[:, 0] + layover_model.cos * near_points[:, 2] / ga_model.tan,
    near_points[:, 1] + layover_model.sin * near_points[:, 2] / ga_model.tan
], axis=1)

# マイナス値をオフセットしてすべて正の値にする
groud_layover_points[:, 0] -= groud_layover_points[:, 0].min()
groud_layover_points[:, 1] -= groud_layover_points[:, 1].min()

x_max = groud_layover_points[:, 0].max()
y_max = groud_layover_points[:, 1].max()

value_map = np.zeros((int(x_max / canvas_span) + (buffer * 2), int(y_max / canvas_span) + (buffer * 2)))

groud_layover_point_cells = (groud_layover_points / canvas_span).astype(int) + buffer

# NumPyのインデックスを利用して value_map を一括更新
np.add.at(value_map, (groud_layover_point_cells[:, 0], groud_layover_point_cells[:, 1]), 10)

# value_map = zoom(value_map, zoom=0.5)
plt.imshow(np.flipud(value_map.T), cmap='gray', origin='lower')

# background color
plt.gca().set_facecolor('gray')
    
# set aspect ratio
plt.gca().set_aspect('equal', adjustable='box')
    
plt.show()