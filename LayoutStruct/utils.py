import numpy as np
from shapely.geometry import Polygon

def svd(A):
    u, s, vh = np.linalg.svd(A)
    S = np.zeros(A.shape)
    S[:s.shape[0], :s.shape[0]] = np.diag(s)
    return u, S, vh


def fit_plane_LSE(points):
    # points: Nx4 homogeneous 3d points
    # return: 1d array of four elements [a, b, c, d] of
    # ax+by+cz+d = 0
    assert points.shape[0] >= 3  # at least 3 points needed
    U, S, Vt = svd(points)
    null_space = Vt[-1, :]
    return null_space

def poly_2d_np2shapely(poly_np):
    # x, y, z = poly_np[:, 0], poly_np[:, 1], poly_np[:, 2]
    # poly_shapely = Polygon(x, y, z)
    poly_shapely = Polygon([tuple(poly_vertex) for poly_vertex in poly_np])
    return poly_shapely

def project_poly2plane_2d(poly, plane, norm=False):
    polygon_xyz = np.array(poly)

    dists_to_plane = polygon_xyz.dot(plane[:3][:, None]) + plane[3]
    poly_proj = polygon_xyz - dists_to_plane * plane[:3][None, :]

    # polygon_xy = polygon_xyz[:, :2] / polygon_xyz[:, 2][:, None]
    # polygon_xy = poly_proj[:, ::2] / poly_proj[:, 1][:, None]

    plane_normal_dir = np.argmax(np.abs(plane[:3]))
    if plane_normal_dir == 0:
        polygon_xy = poly_proj[:, 1:]
        if norm:
            polygon_xy /= polygon_xy[:, 0][:, None] + 1e-4
    elif plane_normal_dir == 1:
        polygon_xy = poly_proj[:, ::2]
        if norm:
            polygon_xy /= polygon_xy[:, 1][:, None] + 1e-4
    elif plane_normal_dir == 2:
        polygon_xy = poly_proj[:, :2]
        if norm:
            polygon_xy /= polygon_xy[:, 2][:, None] + 1e-4

    return polygon_xy

def project_points_to_plane_xy(points, plane, norm=False):
    dists_to_plane = points.dot(plane[:3][:, None]) + plane[3]
    pcd_proj = points - dists_to_plane * plane[:3][None, :]

    # pcd_proj_xy = pcd_proj[:, ::2] / pcd_proj[:, 1][:, None]
    # pcd_proj_xy = pcd_proj[:, ::2]

    plane_normal_dir = np.argmax(np.abs(plane[:3]))
    if plane_normal_dir == 0:
        pcd_proj_xy = pcd_proj[:, 1:]
        if norm:
            pcd_proj_xy /= pcd_proj[:, 0][:, None] + 1e-4
    elif plane_normal_dir == 1:
        pcd_proj_xy = pcd_proj[:, ::2]
        if norm:
            pcd_proj_xy /= pcd_proj[:, 1][:, None] + 1e-4
    elif plane_normal_dir == 2:
        pcd_proj_xy = pcd_proj[:, :2]
        if norm:
            pcd_proj_xy /= pcd_proj[:, 2][:, None] + 1e-4

    return pcd_proj_xy, dists_to_plane