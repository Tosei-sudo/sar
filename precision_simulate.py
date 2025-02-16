# coding: utf-8
import numpy as np

from model.surface import Surface
from model.envioron import Envioron
from model.functions import compute_reflection_vector
from model.simulator import _get_intersection_map, _reproduction_to_ground

import matplotlib.pyplot as plt

surfaces_point = [
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
    # [
    #     [20, 20, 0],
    #     [-20, 20, 0],
    #     [-20, -20, 0],
    #     [20, -20, 0],
    #     [20, 20, 0],
    # ],
]

# ハイパーパラメータ設定
ga = 45
shadow_angle = 0
layover_angle = 245

x_span = 0.1
y_span = 0.1

canvas_span = 0.1

# 環境をモデル化
env = Envioron(ga, shadow_angle, layover_angle, x_span, y_span, canvas_span)

x_range = [np.inf, -np.inf]
y_range = [np.inf, -np.inf]
z_range = [np.inf, -np.inf]

surfaces = []

for surface_points in surfaces_point:
    surface = np.array(surface_points)
    
    x_range = [min(x_range[0], surface[:, 0].min()), max(x_range[1], surface[:, 0].max())]
    y_range = [min(y_range[0], surface[:, 1].min()), max(y_range[1], surface[:, 1].max())]
    z_range = [min(z_range[0], surface[:, 2].min()), max(z_range[1], surface[:, 2].max())]

    sur = Surface(np.array(surface, dtype=np.float))
    surfaces.append(sur)

buffer_norm = int(z_range[1] / env.grazing.tan) + 1
buffer_norm = int(z_range[1] / env.grazing.tan) + 1
buffer = int(max(
    buffer_norm * env.shadow.cos,
    buffer_norm * env.shadow.sin,
    buffer_norm * env.layover.cos,
    buffer_norm * env.layover.sin
))

x_list = np.arange(x_range[0] - buffer, x_range[1] + buffer, x_span)
y_list = np.arange(y_range[0] - buffer, y_range[1] + buffer, y_span)

end_points = np.array([[x, y, 0] for x in x_list for y in y_list])

# 精密計算の実装　ビームの視点を一点に設定
s_range = 2e8

start_points = np.zeros_like(end_points)
start_points[:, 0] = np.mean(x_range) - env.shadow.cos * s_range
start_points[:, 1] = np.mean(y_range) - env.shadow.sin * s_range
start_points[:, 2] = env.grazing.tan * s_range

distance_map, intersection_map = _get_intersection_map(env, surfaces, start_points, end_points)
base_value, x_offset, y_offset = _reproduction_to_ground(env, start_points, end_points, distance_map, intersection_map, buffer)

# 実装メモ
'''
ビームごとにどのサーフェイスで衝突したかを判定
サーフェイスへの入射角を計算し、反射角を計算
衝突した点を始点とし、適当なノルム距離を設定し、終点を計算し、配列に格納
'''

new_start_points = []
new_end_points = []

for (index, start_point) in enumerate(start_points):
    mask = np.all(distance_map[index] == np.inf)

    if not mask:
        # いずれかのサーフェイスでビームが終了した場合
        
        near_point_indices = np.argmin(distance_map[index])
        
        sur = surfaces[near_point_indices]
        sur_normal = np.copy(sur.normal_vector)
        
        new_start_point = intersection_map[index, near_point_indices]
    else:
        # 地表面でビームが終了した場合
        
        sur_normal = np.array([0, 0, 1])
        new_start_point = end_points[index]

    beam_vector = (end_points[index] - start_point)
    beam_vector = -1 * beam_vector / np.linalg.norm(beam_vector)

    new_beam_vector = compute_reflection_vector(beam_vector, sur_normal)
    new_beam_vector = -1 * new_beam_vector / np.linalg.norm(new_beam_vector)
    
    new_start_point = new_start_point + new_beam_vector
    new_end_point = new_start_point + new_beam_vector * 1000

    new_start_points.append(new_start_point)
    new_end_points.append(new_end_point)

new_start_points = np.array(new_start_points)
new_end_points = np.array(new_end_points)

distance_map_2, intersection_map_2 = _get_intersection_map(env, surfaces, new_start_points, new_end_points)

mask = distance_map_2 == np.inf
intersection_map_2 = np.where(mask[:, :, None], intersection_map_2, np.inf)

marged_start_points = np.vstack([start_points, new_start_points])
marged_end_points = np.vstack([end_points, new_end_points])
marged_distance_map = np.vstack([distance_map, distance_map_2])
marged_intersection_map = np.vstack([intersection_map, intersection_map_2])

offset = ((np.array([y_offset, x_offset]) / env.canvas_span).astype(int) + buffer)

value_map, x_offset, y_offset = _reproduction_to_ground(env, marged_start_points, marged_end_points, marged_distance_map, marged_intersection_map, buffer)

offset = offset + - 1 * ((np.array([y_offset, x_offset]) / env.canvas_span).astype(int) + buffer)

value_map = value_map[offset[0] : base_value.shape[0] + offset[0], offset[1] : base_value.shape[1] + offset[1]]

# logを使って平滑化
# value_map = np.log(value_map + 1)
# value_map = value_map ** 2

plt.gca().set_facecolor('gray')
plt.gca().set_aspect('equal', adjustable='box')

plt.imshow(np.flipud(value_map), cmap='gray', origin='lower')
plt.show()

# plt.imsave('result/ground-result.png', np.flipud(value_map.T), cmap='gray', origin='lower')