# coding: utf-8
import math
import random
import numpy as np

from model.functions import get_inner_prod, has_points_in_surface

import uuid

class Surface:
    def __init__(self, points):
        self.points = points
        
        self.uid = uuid.uuid4().hex
        
        self.normal_vector = self.normal()
        self.a_vector = self.get_random_normal()
        
        # 射影用のベクトルを計算
        u = self.a_vector - (np.dot(self.a_vector, self.normal_vector) * self.normal_vector)
        self.u_vector = u / np.linalg.norm(u)
        
        self.v_vector = np.cross(self.normal_vector, self.u_vector)
        
        self.surface_2d = self.get_surface_2d()
    
    def normal(self):
        # 3点を選ぶ
        p0 = self.points[0]
        p1 = self.points[1]
        p2 = self.points[2]
        
        # 2つのベクトルを計算
        v1 = p1 - p0
        v2 = p2 - p0
        
        # 外積を計算
        n = np.cross(v1, v2)
        
        # 正規化
        n = n / np.linalg.norm(n)
        
        return n
    
    def get_random_normal(self):
        while True:
            a = np.array([
                random.randint(0, 100),
                random.randint(0, 100),
                random.randint(0, 100),
            ])
            
            a = a / np.linalg.norm(a)
            
            if np.dot(a, self.normal_vector) != 0:
                break
        
        return a
    
    def get_surface_2d(self):
        return np.array([
            [np.dot(point, self.u_vector), np.dot(point, self.v_vector)] for point in self.points
        ])
    
    def get_intersection_points(self, start_points, end_points, epsilon=1e-6):
        n = self.normal()
        
        beams = end_points - start_points
        d_s = get_inner_prod(np.repeat(n[None, :], beams.shape[0], axis=0), beams)
        
        d_s_mask = d_s != 0
        
        if not np.any(d_s_mask):
            return None, np.repeat(False, len(start_points))
        
        d_s_filtered = d_s[d_s_mask]
        t_s = - (get_inner_prod(np.repeat(n[None, :], d_s_filtered.shape[0], axis=0), start_points[d_s_mask]) - np.dot(n, self.points[0])) / d_s_filtered

        t_s_mask = (t_s >= 0) & (t_s <= 1)
        
        if not np.any(t_s_mask):
            return None, np.repeat(False, len(start_points))

        intersection_points = start_points[d_s_mask][t_s_mask] + t_s[t_s_mask][:, None] * beams[d_s_mask][t_s_mask]

        intersection_points_2d = np.array([
            get_inner_prod(intersection_points, np.repeat(self.u_vector[None, :], len(intersection_points), axis=0)),
            get_inner_prod(intersection_points, np.repeat(self.v_vector[None, :], len(intersection_points), axis=0)),
        ]).T
        
        intersection_mask = has_points_in_surface(self.surface_2d, intersection_points_2d)
        
        not_online_mask = np.linalg.norm(intersection_points - start_points[d_s_mask][t_s_mask], axis=1) > epsilon
        and_mask = intersection_mask & not_online_mask
        
        filtered_intersection_points = intersection_points[and_mask]
        
        result_indexs = np.arange(len(start_points))
        result_indexs = result_indexs[d_s_mask][t_s_mask][and_mask]
        
        return filtered_intersection_points, result_indexs