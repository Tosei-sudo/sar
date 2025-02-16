# coding: utf-8
import numpy as np

class ReproductionMode:
    """ReproductionModeの列挙型
    
    Attributes:
        CONSTANT (int): 定数
        DISTANCE (int): 距離に比例
    """
    CONSTANT = 0
    DISTANCE = 1

def _get_intersection_map(env, surfaces, start_points, end_points):
    distance_map = np.zeros((len(start_points), len(surfaces)))
    distance_map.fill(np.inf)

    intersection_map = np.zeros((len(start_points), len(surfaces), 3))

    for (index, surface) in enumerate(surfaces):
        
        points, indexs = surface.get_intersection_points(start_points, end_points, env.beam_span)
        
        if points is None:
            continue
        
        distance = np.repeat(np.inf, len(start_points))
        
        d = np.linalg.norm(points - start_points[indexs], axis=1)
        distance[indexs] = d

        distance_map[:, index] = distance

        intersection_map[indexs, index] = points
    
    return distance_map, intersection_map

def _reproduction_to_ground(env, receive_point, distance_map, intersection_map):
    """地面に投影したビームの画像化をシミュレートする関数
    
    """
    # 実装メモ
    """
    実際にわかる情報は、
    ・アンテナの位置    receive_point
    ・サーフェイスを貫通したと仮定したビームが衝突した地点までにかかった距離    distance_map
    ・サーフェイスを貫通したと仮定したビームが衝突した地点  intersection_map
    である。
    
    near_points : 実際にビームが衝突した地点
    
    衝突地点の画像上の座標を求めるためには、
    受信地点からビームが衝突した地点までの経過距離を求め、受信した角度を考慮する
    
    beam_distance：ビームが衝突した地点までの経過距離
    """
    import math
    squint = math.radians(60)
    
    # R = np.array([[np.cos(squint), np.sin(squint)], [-np.sin(squint), np.cos(squint)]])
    # flat_cell_coordinates = np.dot(flat_cell_coordinates, R)
    # 条件を満たすインデックスを取得
    inf_mask = np.all(distance_map == np.inf, axis=1)

    # 条件を満たさない部分の処理
    valid_indices = ~inf_mask
    near_point_indices = np.argmin(distance_map[valid_indices], axis=1)
    near_points = intersection_map[valid_indices, near_point_indices]
    
    beam_distance = distance_map[valid_indices, near_point_indices] + np.linalg.norm(near_points - receive_point, axis=1)
    
    range_angle = np.arctan2(near_points[:, 1] - receive_point[1], near_points[:, 0] - receive_point[0]) 
    azimuth_angle = np.arctan2(near_points[:, 2] - receive_point[2], beam_distance)
    
    flat_cell_coordinates = np.array([
        np.cos(azimuth_angle) * beam_distance,
        np.sin(range_angle) * beam_distance
    ]).T
    
    # flat_cell_coordinates[:, 0] *= math.cos(squint)
    flat_cell_coordinates[:, 1] += flat_cell_coordinates[:, 0] * math.sin(squint)
    
    # flat_cell_coordinates = np.array([range_angle * beam_distance, azimuth_angle * beam_distance]).T
    
    range_list = np.min(flat_cell_coordinates, axis=0)
    
    flat_cell_coordinates = flat_cell_coordinates - range_list
    
    image_cell_coordinates = (flat_cell_coordinates / env.canvas_span).astype(int)
    
    value_map = np.zeros((image_cell_coordinates[:, 0].max() + 1, image_cell_coordinates[:, 1].max() + 1))
    
    np.add.at(value_map, (image_cell_coordinates[:, 0], image_cell_coordinates[:, 1]), 10)
    
    return value_map