# coding: utf-8
import math
import random
import numpy as np

from model.functions import get_inner_prod, has_points_in_surface

import uuid

class Surface:
    def __init__(self, points):
        """
        Surface クラスのコンストラクタ。
        ポリゴンの頂点情報を受け取り、法線ベクトルや射影用のベクトルを計算する。
        """
        self.points = points  # ポリゴンの頂点座標
        
        self.uid = uuid.uuid4().hex  # 一意の識別子を生成
        
        self.normal_vector = self.normal()  # 法線ベクトルを計算
        self.a_vector = self.get_random_normal()  # 法線と直交しないランダムなベクトルを生成
        
        # 射影用の基底ベクトルを計算
        u = self.a_vector - (np.dot(self.a_vector, self.normal_vector) * self.normal_vector)
        self.u_vector = u / np.linalg.norm(u)  # 平面内の単位ベクトル u
        self.v_vector = np.cross(self.normal_vector, self.u_vector)  # u と直交する v を求める
        
        self.surface_2d = self.get_surface_2d()  # 2D 平面上のポリゴンを計算
    
    def normal(self):
        """
        平面の法線ベクトルを計算する。
        """
        # 3点を選ぶ（ポリゴンの最初の3点）
        p0 = self.points[0]
        p1 = self.points[1]
        p2 = self.points[2]
        
        # 2つのベクトルを計算
        v1 = p1 - p0
        v2 = p2 - p0
        
        # 外積を計算して法線ベクトルを求める
        n = np.cross(v1, v2)
        
        # 正規化（単位ベクトル化）
        n = n / np.linalg.norm(n)
        
        return n
    
    def get_random_normal(self):
        """
        法線ベクトルと平行でないランダムなベクトルを生成する。
        """
        while True:
            # ランダムなベクトルを生成
            a = np.array([
                random.randint(0, 100),
                random.randint(0, 100),
                random.randint(0, 100),
            ])
            
            a = a / np.linalg.norm(a)  # 正規化
            
            # 法線ベクトルと平行でないか確認
            if np.dot(a, self.normal_vector) != 0:
                break
        
        return a
    
    def get_surface_2d(self):
        """
        平面の3D座標を2Dに射影する。
        """
        return np.array([
            [np.dot(point, self.u_vector), np.dot(point, self.v_vector)] for point in self.points
        ])
    
    def get_intersection_points(self, start_points, end_points, epsilon=1e-6):
        """
        レイ（start_points → end_points）と平面の交点を求める。
        """
        n = self.normal()  # 法線ベクトル
        
        # レイの方向ベクトル
        beams = end_points - start_points
        
        # 法線とレイの方向ベクトルの内積を計算（交差判定用）
        d_s = get_inner_prod(np.repeat(n[None, :], beams.shape[0], axis=0), beams)
        
        # d_s が 0 の場合、平面と平行で交点なし
        d_s_mask = d_s != 0
        
        if not np.any(d_s_mask):
            return None, np.repeat(False, len(start_points))
        
        d_s_filtered = d_s[d_s_mask]  # 平面と交差するビームのみ抽出
        
        # 交点のパラメータ t_s を計算
        t_s = - (get_inner_prod(np.repeat(n[None, :], d_s_filtered.shape[0], axis=0), start_points[d_s_mask]) - np.dot(n, self.points[0])) / d_s_filtered

        # 交点がレイの範囲内（0 <= t_s <= 1）のみ有効
        t_s_mask = (t_s >= 0) & (t_s <= 1)
        
        if not np.any(t_s_mask):
            return None, np.repeat(False, len(start_points))

        # 交点を計算
        intersection_points = start_points[d_s_mask][t_s_mask] + t_s[t_s_mask][:, None] * beams[d_s_mask][t_s_mask]

        # 交点を 2D 平面に射影
        intersection_points_2d = np.array([
            get_inner_prod(intersection_points, np.repeat(self.u_vector[None, :], len(intersection_points), axis=0)),
            get_inner_prod(intersection_points, np.repeat(self.v_vector[None, :], len(intersection_points), axis=0)),
        ]).T
        
        # 交点がポリゴンの内部にあるか判定
        intersection_mask = has_points_in_surface(self.surface_2d, intersection_points_2d)
        
        # 交点がレイの始点と異なるか判定（数値誤差を考慮）
        not_online_mask = np.linalg.norm(intersection_points - start_points[d_s_mask][t_s_mask], axis=1) > epsilon
        and_mask = intersection_mask & not_online_mask
        
        # 有効な交点を抽出
        filtered_intersection_points = intersection_points[and_mask]
        
        # 交点の元のインデックスを取得
        result_indexs = np.arange(len(start_points))
        result_indexs = result_indexs[d_s_mask][t_s_mask][and_mask]
        
        return filtered_intersection_points, result_indexs
