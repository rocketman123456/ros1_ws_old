import numpy as np
from evaluator import Evaluator
from visualizer import draw_bb_planes, draw_compare, draw_planes, draw_voxel_correspondence
from fileio import IOHelper
import open3d as o3d
import sys

# TODO load cloud once, compare all algos at once?


def evaluate(cloud_path: str, gt_path: str, algo_path: str, debug=False, voxelsize = 0.13) -> None:
    iohelper = IOHelper(cloud_path, gt_path, algo_path)

    if np.any(points := iohelper.read_cloud()):
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(points)
        # pointcloud.colors = o3d.utility.Vector3dVector(colors)
    else:
        pointcloud = iohelper.read_pcd()
    # colors  = np.loadtxt(cloud_path, dtype=int, usecols=(3,4,5))
    # colors  = colors * (1/255)
    ground_truth = iohelper.read_gt()
    test = iohelper.read_algo()
    if debug:
        draw_compare(ground_truth, test, pointcloud)
    if len(test)== 0:
        return
    # if 3dkht, translate algo_planes by pcd_bb center
    if iohelper.method == '3DKHT':
        print('3DKHT, translating')
        for algo_plane in test:
            algo_plane.translate(pointcloud.get_center())

    if debug:
        draw_planes(test, pointcloud)

    # draw_planes(test, pointcloud)

    if debug:
        draw_planes(test, pointcloud)

    kdtree = o3d.geometry.KDTreeFlann(pointcloud)
    if debug:
        o3d.visualization.draw_geometries([pointcloud])

    # if own datasets, find corresponding indices for planes in xyz format
    if ground_truth[0].indices == []:
        for plane in ground_truth:
            plane.get_indices_from_kdtree(kdtree)

    if test[0].indices == []:
        for plane in test:
            plane.get_indices_from_kdtree(kdtree)

    octree = o3d.geometry.Octree(max_depth=8)
    octree.convert_from_point_cloud(pointcloud)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pointcloud, voxel_size=voxelsize)

    if debug:
        o3d.visualization.draw_geometries([octree])

    # inlier_evaluator = Evaluator.create(points, ground_truth, test)
    # print('calculating correspondence')
    # inlier_evaluator.correspondence()
    # print('done calculating correspondence')
    # p, r, f1 = inlier_evaluator.get_metrics()
    # iohelper.save_results(p,r,f1)
    # print(f'{inlier_evaluator.precision = }')
    # print(f'{inlier_evaluator.recall = }')
    # print(f'{inlier_evaluator.f1 = }')
    # f = set()
    # for i, k in inlier_evaluator.correspondences.items():
    #     if k != None:
    #         f.add(k)
    # print(f'found: {len(f)} / {len(ground_truth)}')
    # input("press enter to continue with voxel evaluation")

    voxel_evaluator = Evaluator.create(points, ground_truth, test, voxel_grid)
    print('calculating correspondence')
    voxel_evaluator.correspondence()
    if debug:
        draw_voxel_correspondence(ground_truth, test, pointcloud)
    print('done calculating correspondence')
    voxel_evaluator.calc_voxels(pointcloud)
    p, r, f1 = voxel_evaluator.get_metrics()
    print(p, r, f1)
    f = set()
    for gtp in voxel_evaluator.correspondences.values():
        if gtp != None:
            f.add(gtp)

    total, per_plane, per_sample = iohelper.get_times()

    iohelper.save_results(p, r, f1, len(f), len(
        ground_truth), total, per_plane, per_sample)
    # print(f'found: {len(f)} / {len(ground_truth)}')
    # input("press enter to continue with octree evaluation")

    # octree_evaluator = Evaluator.create(points, ground_truth, test, octree)
    # print('calculating correspondence')
    # octree_evaluator.correspondence()
    # print('calculating leaf nodes')
    # p, r, f1 = octree_evaluator.get_metrics()
    # iohelper.save_results(p,r,f1)

    # print(f'{octree_evaluator.precision = }')
    # print(f'{octree_evaluator.recall = }')
    # print(f'{octree_evaluator.f1 = }')
    # f = set()
    # for i, k in octree_evaluator.correspondences.items():
    #     if k != None:
    #         f.add(k)
    # print(f'found: {len(f)} / {len(ground_truth)}')


if __name__ == '__main__':

    # cloud_path = sys.argv[1]
    # gt_path = sys.argv[2]
    # algo_path = sys.argv[3]
    cloud_path = "Stanford3dDataset_v1.2_Aligned_Version/TEST/hallway_7/hallway_7.txt"
    gt_path = "Stanford3dDataset_v1.2_Aligned_Version/TEST/hallway_7/GT"
    algo_path = "Stanford3dDataset_v1.2_Aligned_Version/TEST/hallway_7/RSPyD"

    # cloud_path = "/home/pedda/Documents/uni/BA/clones/datasets/RSPD/pointclouds/boiler_room.pcl"
    # gt_path = "/home/pedda/Documents/uni/BA/clones/datasets/RSPD/detections/boiler_room_ground_truth.geo"
    # algo_path = "/home/pedda/Documents/uni/BA/clones/datasets/RSPD/detections/boiler_room_ransac_schnabel.geo"
    # cloud_path = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/Stanford3dDataset_v1.2_Aligned_Version/Area_2/auditorium_2/auditorium_2.txt"
    # gt_path = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/Stanford3dDataset_v1.2_Aligned_Version/Area_2/auditorium_2/GT"
    # algo_path = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/Stanford3dDataset_v1.2_Aligned_Version/Area_2/auditorium_2/RSPD/auditorium_2.geo"
    # cloud_path = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/FIN-Dataset/auditorium/1664003770.004746437.txt"
    # gt_path = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/FIN-Dataset/auditorium/GT"
    # algo_path = "/home/pedda/Documents/uni/BA/clones/PlaneDetection/CommandLineOption/test/"
    # algo_path = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/Stanford3dDataset_v1.2_Aligned_Version/Area_1//RSPD"
    evaluate(cloud_path, gt_path, algo_path)
