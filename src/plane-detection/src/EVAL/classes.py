
from copy import deepcopy
from dataclasses import dataclass
import os
from typing import List
import open3d as o3d
import numpy as np


@dataclass
class Result():
    precision: float
    recall: float
    f1: float
    detected: int
    out_of: int
    dataset: str
    algorithm: str
    time_total: float
    time_per_plane: float
    time_per_sample: float

    def to_file(self, path: str):
        print(f'Writing results to {path}')
        with open(path, 'w') as ofile:
            ofile.write(f'{self.algorithm} : {self.dataset} \n')
            ofile.write(f'precision: {self.precision}\n')
            ofile.write(f'recall: {self.recall}\n')
            ofile.write(f'f1-score: {self.f1}\n')
            ofile.write(f'found: {self.detected} / {self.out_of}\n')
            ofile.write('pre calc post\n')
            ofile.write(
                f'{self.time_total} {self.time_per_plane} {self.time_per_sample}')

    @staticmethod
    def from_file(path: str):
        algo, dataset = np.loadtxt(
            path, dtype=str, usecols=(0, 2), max_rows=1)
        prec, rec, f1 = np.loadtxt(
            path, dtype=float, skiprows=1, usecols=1, max_rows=3)
        detected, out_of = np.loadtxt(
            path, dtype=int, usecols=(1, 3), skiprows=4, max_rows=1)
        total, per_plane, per_sample = np.loadtxt(
            path, dtype=float, skiprows=6)
        return Result(
            precision=prec,
            recall=rec,
            f1=f1,
            detected=int(detected),
            out_of=int(out_of),
            dataset=dataset,
            algorithm=algo,
            time_total=total,
            time_per_plane=per_plane,
            time_per_sample=per_sample
        )


def through_crop(plane, subcloud: o3d.geometry.PointCloud, voxel_grid: o3d.geometry.VoxelGrid, complete_cloud: o3d.geometry.PointCloud):

    # iterate over points in ground truth
    # remove all that are not included or that have no close neighbors
    sub_tree = o3d.geometry.KDTreeFlann(subcloud)

    crop_plane = Plane()
    for point in plane.xyz_points:
        [k, idx, A] = sub_tree.search_radius_vector_3d(point, 0.05)
        if k > 0:
            crop_plane.xyz_points.append(point)
            for i in idx:
                crop_plane.set_indices.add(i)
                crop_plane.indices.append(i)
    print(len(crop_plane.xyz_points))
    print(len(crop_plane.set_indices))
    # if len(crop_plane.xyz_points) > 3:
    #     pts = o3d.utility.Vector3dVector(crop_plane.xyz_points)
    #     pc = o3d.geometry.PointCloud()
    #     pc.points = pts
    #     obb = pc.get_oriented_bounding_box()
    #     vgdense = o3d.geometry.VoxelGrid.create_dense(obb.center,np.random.rand(3),0.4,  obb.extent[0], obb.extent[1],obb.extent[2] )
    #     dvoxels = vgdense.get_voxels()
    #     vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, 0.4)
    #     voxel = vg.get_voxels()
    #     if len(dvoxels) == 0 or len(voxel) / len(dvoxels) < 0.50:
    #         return None
    return crop_plane

class Plane:
    def __init__(self) -> None:
        self.indices: List[int] = []
        self.name: str = ""
        self.xyz_points = []
        self.kd = None
        self.set_indices = set()
        self.set_points = set()
        self.voxels = set()
        self.normal = []
        self.leafs = set()

    def translate(self, vector: np.ndarray):
        if len(self.xyz_points) == 0:
            return
        if isinstance(self.xyz_points[0], float):
            self.xyz_points = [self.xyz_points]
        for point in self.xyz_points:
            point[0] += vector[0]
            point[1] += vector[1]
            point[2] += vector[2]

    def calc_voxel(self, vg: o3d.geometry.VoxelGrid, pointcloud: o3d.geometry.PointCloud):
        if self.xyz_points == []:
            self.calc_xyz(pointcloud)
        for inlier in self.xyz_points:
            v = vg.get_voxel(inlier)
            self.voxels.add(tuple(v))

    def get_indices_from_kdtree(self, kdtree: o3d.geometry.KDTreeFlann):
        for point in self.xyz_points:
            [k, idx, _] = kdtree.search_knn_vector_3d(point, 1)
            self.indices.append(idx[0])
            self.set_indices.add(idx[0])

    def calc_xyz(self, pointcloud: o3d.geometry.PointCloud):
        for point in self.set_indices:
            self.xyz_points.append(pointcloud.points[point])

    def set_set(self, pc: o3d.geometry.PointCloud):
        # self.set_indices = set(self.indices)
        for i in self.set_indices:
            self.set_points.add(tuple(pc.points[i]))

    @staticmethod
    def xyzfrom_txt(file: str):
        points = np.loadtxt(file, usecols=(0, 1, 2)).tolist()
        p = Plane()
        p.xyz_points = points
        p.name = file.split('/')[-1]
        return p

    @staticmethod
    def i_from_txt(file: str):
        points = np.loadtxt(file, usecols=(0)).tolist()
        p = Plane()
        p.indices = points
        p.name = file.split('/')[-1]
        return p


class NULL(Plane):
    def __init__(self) -> None:
        self.name = "NULL"

    @staticmethod
    def create():
        return NULL()
