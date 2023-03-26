import abc
from typing import Dict, List
import open3d as o3d
import numpy as np
from tqdm import tqdm

from classes import Plane


class Evaluator(abc.ABC):
    @staticmethod
    def create(points: np.ndarray, ground_truth: List[Plane], test: List[Plane], method=None) -> "OctreeEvaluator | InlierEvaluator | VoxelEvaluator":
        if method == None:
            return InlierEvaluator(points, ground_truth, test)
        elif isinstance(method, o3d.geometry.Octree):
            return OctreeEvaluator(points, ground_truth, test, method)
        elif isinstance(method, o3d.geometry.VoxelGrid):
            print("creating VoxelGridEvaluator")
            return VoxelEvaluator(points, ground_truth, test, method)

    def calc_voxels(self, pointcloud):
        pass

    def get_metrics(self):
        self.get_precision()
        self.get_recall()
        self.get_f1()
        return self.precision, self.recall, self.f1

    def get_corr(self, plane: Plane):
        count = dict()
        count[None] = 0
        for inlier in plane.indices:
            found = False
            for g in self.ground_truth:
                if inlier in g.set_indices:
                    if g not in count.keys():
                        count[g] = 0
                    count[g] += 1
                    found = True
                    break
            if not found:
                count[None] += 1
        best = None

        for p, i in count.items():
            if i > count[best]:
                best = p
        if best == None or count[best] < len(plane.indices) / 2:
            return None
        return best

    def correspondence(self):
        self.test: List[Plane]
        self.ground_truth: List[Plane]
        self.correspondences: Dict[Plane, "Plane | None"] = dict()
        for plane in tqdm(self.test):
            ret = self.get_corr(plane)
            self.correspondences[plane] = ret
        
    @abc.abstractmethod
    def get_precision(self) -> float:
        pass

    @abc.abstractmethod
    def get_recall(self) -> float:
        pass

    def get_f1(self) -> float:
        if self.precision == 0.0 and self.recall == 0.0:
            self.f1 = 0.0
            return self.f1
        self.f1 = 2*(self.precision * self.recall) / \
            (self.precision + self.recall)
        return self.f1


class OctreeEvaluator(Evaluator):
    def __init__(self, points: np.ndarray, ground_truth: List[Plane], test: List[Plane], octree) -> None:
        self.points = points
        self.ground_truth = ground_truth
        self.test = test
        self.octree: o3d.geometry.Octree = octree

    def calc_leafs(self, pc: o3d.geometry.PointCloud):
        for plane in tqdm(self.ground_truth):
            for inlier in plane.set_indices:
                _, info = self.octree.locate_leaf_node(pc.points[inlier])
                plane.leafs.add(tuple(info.origin))
        for plane in tqdm(self.test):
            for inlier in plane.set_indices:
                _, info = self.octree.locate_leaf_node(pc.points[inlier])
                plane.leafs.add(tuple(info.origin))

    def get_precision(self):
        self.all = set()
        self.correct = set()

        print(f'{len(self.correspondences.items()) = }')
        for test, gt in self.correspondences.items():
            for leaf in test.leafs:
                # inlier = np.array(pc.points[test_inlier])
                # leaf, info = self.octree.locate_leaf_node(inlier)
                # self.all.add(hash(tuple(info.origin)))
                self.all.add(leaf)
                if gt != None and leaf in gt.leafs:
                    self.correct.add(leaf)
        print(len(self.correct), '/', len(self.all))
        self.precision = len(self.correct) / len(self.all)

    def get_recall(self):
        self.all = set()
        self.correct = set()
        for gp in self.ground_truth:
            for leaf in gp.leafs:
                self.all.add(leaf)
                for t, g in self.correspondences.items():
                    if g == gp and leaf in t.leafs:
                        self.correct.add(leaf)
                        break
        print(f'{len(self.correct)} / {len(self.all)}')
        self.recall = len(self.correct) / len(self.all)
        return len(self.correct) / len(self.all)


class InlierEvaluator(Evaluator):
    def __init__(self, points: np.ndarray, ground_truth: List[Plane], test: List[Plane]) -> None:
        self.points = points
        self.ground_truth = ground_truth
        self.test = test

    def get_precision(self):
        self.all = set()
        self.correct = set()
        for test, gt in self.correspondences.items():
            for test_inlier in test.set_indices:
                self.all.add(test_inlier)
                if gt != None and test_inlier in gt.set_indices:
                    self.correct.add(test_inlier)
        self.precision = len(self.correct) / len(self.all)

    def get_recall(self):
        al = set()
        correct = set()
        for gp in self.ground_truth:
            for inlier in gp.set_indices:
                al.add(inlier)
                for x, y in self.correspondences.items():
                    if y == gp and inlier in x.set_indices:
                        correct.add(inlier)
                        break
        print(f'{len(correct)} / {len(al)}')
        self.recall = len(correct) / len(al)
        return len(correct) / len(al)


class VoxelEvaluator(Evaluator):
    def __init__(self, points: np.ndarray, ground_truth: List[Plane], test: List[Plane], voxel_grid) -> None:
        self.points = points
        self.ground_truth = ground_truth
        self.test = test
        self.voxel_grid = voxel_grid

    def calc_voxels(self, pc: o3d.geometry.PointCloud):
        for plane in self.ground_truth:
            for inlier in plane.set_indices:
                v = self.voxel_grid.get_voxel(pc.points[inlier])
                plane.voxels.add(tuple(v))
        for plane in self.test:
            for inlier in plane.set_indices:
                v = self.voxel_grid.get_voxel(pc.points[inlier])
                plane.voxels.add(tuple(v))

    def get_precision(self):
        self.all = set()
        self.correct = set()
        for ap, gt in tqdm(self.correspondences.items()):
            for a_v in ap.voxels:
                h_a_v = hash(a_v)
                self.all.add(h_a_v)
                if gt != None and a_v in gt.voxels:
                    self.correct.add(h_a_v)
        if len(self.all) == 0:
            self.precision = 0.0
            return 0.0
        self.precision = len(self.correct) / len(self.all)
        return len(self.correct) / len(self.all)

    def get_recall(self):
        self.all = set()
        self.correct = set()
        for plane in self.ground_truth:
            for voxel in plane.voxels:
                self.all.add(voxel)
                for t, g in self.correspondences.items():
                    if g == plane and voxel in t.voxels:
                        self.correct.add(voxel)
        if len(self.all) == 0:
            self.recall = 0.0
            return 0.0
        self.recall = len(self.correct) / len(self.all)
        return len(self.correct) / len(self.all)
