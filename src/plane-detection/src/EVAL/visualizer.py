from typing import List
import open3d as o3d
from classes import Plane
import numpy as np

def draw_planes(planes: List[Plane], pointcloud):
    clouds  = []
    for plane in planes:
        points = []
        if len(plane.xyz_points) < 1:
            plane.calc_xyz(pointcloud)
        for pts in plane.xyz_points:
            points.append(pts)
        
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.paint_uniform_color(np.random.rand(3))
        clouds.append(cloud)
    o3d.visualization.draw_geometries(clouds)


def draw_compare(gt: List[Plane], test: List[Plane], pointcloud):
    gtpoints  = []
    for plane in gt:
        for pts in plane.xyz_points:
            gtpoints.append(pts)
    gtcloud = o3d.geometry.PointCloud()
    gtcloud.points = o3d.utility.Vector3dVector(gtpoints)
    gtcloud.paint_uniform_color([0, 1,0])
    points  = []
    for plane in test:
        if len(plane.xyz_points) < 1:
            plane.calc_xyz(pointcloud)
        for pts in plane.xyz_points:
            points.append(pts)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.paint_uniform_color([0, 0,1])



    o3d.visualization.draw_geometries([gtcloud, cloud])
    o3d.visualization.draw_geometries([cloud])

def draw_bb_planes(planes: List[Plane], pointcloud : o3d.geometry.PointCloud, test: List[Plane] = None):
    to_draw = [pointcloud]
    for plane in planes:
        cloud = o3d.geometry.PointCloud()
        if len(plane.xyz_points) < 1:
            plane.calc_xyz(pointcloud)
        cloud.points = o3d.utility.Vector3dVector(plane.xyz_points)
        cloud.paint_uniform_color([0.8,0.8,0.8])
        aabb = cloud.get_oriented_bounding_box()
        aabb.color = (0, 1, 0)
        to_draw.append(aabb)
        # to_draw.append(cloud)
    if test:
        for plane in test:
            cloud = o3d.geometry.PointCloud()
            if len(plane.xyz_points) < 1:
                plane.calc_xyz(pointcloud)
            cloud.points = o3d.utility.Vector3dVector(plane.xyz_points)
            cloud.paint_uniform_color([0.8,0.8,0.8])
            aabb = cloud.get_oriented_bounding_box()
            aabb.color = (1, 0, 0)
            to_draw.append(aabb)
            # to_draw.append(cloud)

    o3d.visualization.draw_geometries(to_draw)

def draw_voxel_correspondence(gt: List[Plane], test: List[Plane], pc: o3d.geometry.PointCloud):
    joint_cloud = []
    t_indices = set()
    gt_indices = set()
    for gp in gt:
        gt_indices.update(gp.set_indices)
    for tp in test:
        t_indices.update(tp.set_indices)
    pc.paint_uniform_color([0.8, 0.8, 0.8])
    trues = gt_indices.intersection(t_indices)
    fns = [i for i in gt_indices if i not in t_indices]
    fps = [i for i in t_indices if i not in gt_indices]
    colors = np.full_like(pc.points, np.array([0.8,0.8,0.8]))
    for index in trues: # true : Green
        colors[index] = np.array([0,1,0])
    for index in fns:   # FN  : Red
        colors[index] = np.array([1,0,0])
    for index in fps:   # FP  : B
        colors[index] = np.array([0,0,1])
                    # TN : Gray
    pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pc])

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pc, voxel_size=0.25)
    o3d.visualization.draw_geometries([voxel_grid])