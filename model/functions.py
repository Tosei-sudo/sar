# coding: utf-8
import math
import numpy as np

# 内積を計算する。u_s, v_sはそれぞれのベクトルの集合
def get_inner_prod(u_s, v_s):
    return np.sum(u_s * v_s, axis=1)

def has_points_in_surface(surface, points):
    surface = np.asarray(surface)
    points = np.asarray(points)
    
    # ポリゴンが閉じていなければ閉じる（最初の点を末尾に追加）
    if not np.allclose(surface[0], surface[-1]):
        surface = np.vstack([surface, surface[0]])
    
    # ポリゴンの各エッジを定義（連続する頂点の組）
    start_points = surface[:-1]  # shape: (M, 2)
    end_points   = surface[1:]   # shape: (M, 2)
    
    # 各エッジのベクトルを計算
    edges = start_points - end_points  # shape: (M, 2)
    
    # 各エッジの始点から各点へのベクトルを計算
    # start_points: (M, 2) → (1, M, 2)
    # points: (N, 2) → (N, 1, 2)
    vec = start_points[None, :] - points[:, None, :]  # shape: (N, M, 2)
    
    # 2Dの場合、np.crossは各ペアについてスカラー値（外積の大きさ＆符号）を返す
    # edges: (1, M, 2) と vec: (N, M, 2) の外積 → (N, M)
    cross = np.cross(edges[None, :], vec)
    
    # 各点について、すべてのエッジに対して外積の符号がすべて非負またはすべて非正なら内部と判定
    inside = np.logical_or(np.all(cross >= 0, axis=1),
                           np.all(cross <= 0, axis=1))
    
    return inside

def compute_reflection_vector(incident_vector, normal_vector):
    """
    入射ベクトルと法線ベクトルをもとに、反射ベクトルを計算する関数。

    Parameters:
        incident_vectors (np.ndarray): 入射ベクトル (N, 3) の形状を持つ配列
        normal_vectors (np.ndarray): 法線ベクトル (N, 3) の形状を持つ配列（正規化済み）

    Returns:
        np.ndarray: 反射ベクトル (N, 3) の形状を持つ配列
    """
    return incident_vector - 2 * np.dot(incident_vector, normal_vector) * normal_vector