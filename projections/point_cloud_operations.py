import torch
import numpy as np
from open3d import *
import open3d

def rotate_x(pos):
    num = pos.shape[0]

    for i in range(num):
        a=pos[i][1]
        pos[i][1] = pos[i][2]
        pos[i][2] = -a
        
    return pos
    

def read_pcd(filename, isRefine=False):
    pcd = open3d.read_point_cloud(filename)
    if isRefine:
        pcd = remove_outlier(pcd)
    return np.array(pcd.points)


def remove_outlier(pcd):
    cl, ind = radius_outlier_removal(pcd, nb_points=6, radius=0.002*3*2.0)
    if len(ind) < 500:
        return pcd
    return select_down_sample(pcd, ind)

def save_pcd(filename, points, isRefine=True):
    pcd = PointCloud()
    pcd.points = Vector3dVector(points)

    if isRefine:
        pcd = remove_outlier(pcd)
    #print('ref')
    write_point_cloud(filename, pcd)

def save_pcd_textures(filename, points, colors, isRefine=True):
    pcd = PointCloud()
    pcd.points = Vector3dVector(points)
    pcd.colors = Vector3dVector(colors)

    print(points.shape)
    if isRefine:
        pcd = remove_outlier(pcd)    
    print(np.asarray(pcd.points).shape)
    write_point_cloud(filename, pcd)


def save_pcd_colors(pos, colors, path):
    pcd = PointCloud()
    pcd.points = Vector3dVector(pos)
    pcd.colors = Vector3dVector(colors)

    write_point_cloud(path, pcd)