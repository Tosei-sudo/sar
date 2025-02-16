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

def _reproduction_to_ground(env, start_points, end_points, distance_map, intersection_map, buffer=1, mode=ReproductionMode.CONSTANT):
    """地面に投影したビームの画像化をシミュレートする関数
    
    """
    # 条件を満たすインデックスを取得
    inf_mask = np.all(distance_map == np.inf, axis=1)

    # 条件を満たさない部分の処理
    valid_indices = ~inf_mask
    near_point_indices = np.argmin(distance_map[valid_indices], axis=1)
    near_points = intersection_map[valid_indices, near_point_indices]
    
    # `groud_layover_points` の初期化
    groud_layover_points = np.zeros((len(start_points), 2))

    # 条件を満たす部分を直接代入(ビームがどのサーフェイスにも衝突しない場合)
    # groud_layover_points[inf_mask] = np.array([np.nan, np.nan])
    groud_layover_points[inf_mask] = end_points[inf_mask, :2]
    # end_points[inf_mask, :2]

    groud_layover_points[valid_indices] = np.stack([
        near_points[:, 0] + env.layover.cos * near_points[:, 2] / env.grazing.tan,
        near_points[:, 1] + env.layover.sin * near_points[:, 2] / env.grazing.tan
    ], axis=1)

    # マイナス値をオフセットしてすべて正の値にする
    x_offset = np.ma.masked_invalid(groud_layover_points[:, 0]).min()
    y_offset = np.ma.masked_invalid(groud_layover_points[:, 1]).min()

    groud_layover_points[:, 0] -= x_offset
    groud_layover_points[:, 1] -= y_offset

    # infを覗いた最大値を取得
    x_max = np.ma.masked_invalid(groud_layover_points[:, 0]).max()
    y_max = np.ma.masked_invalid(groud_layover_points[:, 1]).max()
    
    value_map = np.zeros((int(x_max / env.canvas_span) + (buffer * 2), int(y_max / env.canvas_span) + (buffer * 2)))

    groud_layover_point_cells = (groud_layover_points / env.canvas_span).astype(int) + buffer

    # value_mapの範囲から外れるインデックスを取得し、フィルタリング
    groud_layover_point_cells = groud_layover_point_cells[(groud_layover_point_cells[:, 0] >= 0) & (groud_layover_point_cells[:, 0] < value_map.shape[0])]

    # NumPyのインデックスを利用して value_map を一括更新
    np.add.at(value_map, (groud_layover_point_cells[:, 0], groud_layover_point_cells[:, 1]), 10)
    
    return value_map.T, x_offset, y_offset

def simulate(env, surfaces, start_points, end_points, buffer=1, mode=ReproductionMode.DISTANCE):
    """ビームの画像化をシミュレートする関数

    Args:
        env (model.envioron.Envioron): 環境情報
        surfaces (_type_): サーフェイスの配列
        start_points (_type_): ビームの始点
        end_points (_type_): ビームの終点
        buffer (int, optional): バッファ. Defaults to 1.
        mode (ReproductionMode, optional): 再生モード. Defaults to ReproductionMode.DISTANCE.

    Returns:
        ndarray: シミュレート結果
    """
    distance_map, intersection_map = _get_intersection_map(env, surfaces, start_points, end_points)
    
    return _reproduction_to_ground(env, start_points, end_points, distance_map, intersection_map, buffer, mode)[0]