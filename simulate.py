# coding: utf-8
import numpy as np

from model.surface import Surface
from model.envioron import Envioron
from model.simulator import simulate

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
layover_angle = 180

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

# 実装メモ
'''
サーフェイスごとにビームの衝突を判定
交点を記録し、始点に一番近い交点を反射点に選択
反射点をレイオーバーを考慮して2次元に投影する
'''

value_map = simulate(env, surfaces, start_points, end_points, buffer)

plt.gca().set_facecolor('gray')
plt.gca().set_aspect('equal', adjustable='box')

# print(time.time() - start)
plt.imshow(np.flipud(value_map), cmap='gray', origin='lower')
plt.show()

# plt.imsave("result/single-result.png", np.flipud(value_map.T), cmap='gray', origin='lower')