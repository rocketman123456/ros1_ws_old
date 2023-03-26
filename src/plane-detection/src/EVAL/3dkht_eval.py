import argparse
import matplotlib.pyplot as plt
import os
import open3d as o3d
from typing import Dict, List
from classes import Result
from evaluator import Evaluator
from evaluate import evaluate
import pandas as pd
from fileio import IOHelper, create_pcd
import seaborn as sb
import numpy as np
from visualizer import draw_planes
# globals
KHT_PARAMS = [(1, 0.025),(2, 10.0), (4, 0.2), (5, 0.2), (6, 0.3) , (7, 0.5)]
ALGO_IN = {'RSPD': '.txt', 'OPS': '.pcd', '3DKHT': '.txt'}
plt.rcParams.update({'font.size': 22})


def e(cloud_path, gt_path, algo_path):
    iohelper = IOHelper(cloud_path, gt_path, algo_path)
    points = iohelper.read_cloud()
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    ground_truth = iohelper.read_gt()

    for s_l, d2p in KHT_PARAMS:
        test = iohelper.read_kht(s_l)

        print('3DKHT, translating')
        for algo_plane in test:
            algo_plane.translate(pointcloud.get_center())
        print(
            f'params:\t{s_l}, {d2p}\n#planes:\t{len(test)} / {len(ground_truth)}')

        # draw_planes(test, pointcloud)
        kdtree = o3d.geometry.KDTreeFlann(pointcloud)

        # if own datasets, find corresponding indices for planes in xyz format
        if ground_truth[0].indices == []:
            for plane in ground_truth:
                plane.get_indices_from_kdtree(kdtree)

        if len(test) > 0 and test[0].indices == []:
            for plane in test:
                plane.get_indices_from_kdtree(kdtree)

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pointcloud, voxel_size=0.13)

        voxel_evaluator = Evaluator.create(
            points, ground_truth, test, voxel_grid)
        print('calculating correspondence')
        voxel_evaluator.correspondence()

        print('done calculating correspondence')
        voxel_evaluator.calc_voxels(pointcloud)
        if len(test) > 0:
            p, r, f1 = voxel_evaluator.get_metrics()
        else:
            p = r = f1 = 0.0
        f = set()
        for gtp in voxel_evaluator.correspondences.values():
            if gtp != None:
                f.add(gtp)

        total, per_plane, per_sample = iohelper.get_times()

        iohelper.save_kht_results(p, r, f1, len(f), len(
            ground_truth), total, per_plane, per_sample, s_l)


def kht_eval(root_folder: str):
    datasets = os.listdir(root_folder)
    algo = '3DKHT'
    for dataset in datasets:
        # ignore files, results and datasets without GT
        if os.path.isfile(os.path.join(root_folder, dataset)):
            continue
        if dataset.startswith('nope_') or dataset.startswith('results'):
            continue
        dataset_path = os.path.join(root_folder, dataset)
        gt_path = os.path.join(dataset_path, "GT")
        cloud_filename = dataset + '.txt'
        cloud_path = os.path.join(dataset_path, cloud_filename)
        algo_path = os.path.join(dataset_path, algo)
        e(cloud_path, gt_path, algo_path)


def kht_collect(root_folder):
    results_per_scene: Dict[str, List[Result]] = dict()
    # algo -> [scene_type -> [scene_specific_result]]
    results_per_param: Dict[int, List[Result]] = {
        param[0]: [] for param in KHT_PARAMS}
    datasets = os.listdir(root_folder)
    for dataset in datasets:
        if os.path.isfile(os.path.join(root_folder, dataset)):
            # skip files
            continue
        if dataset.startswith('nope_') or dataset == 'results':
            # skip non-datasets (without GT or results directory)
            continue
        results_per_scene[dataset] = []
        scene_type = dataset.split('_')[0]
        result_path = os.path.join(root_folder, dataset, 'results')

        # populate result dicts, per_scene is currently unused, might be useful later
        for algorithm_resultfile in os.listdir(result_path):
            if '3DKHT' not in algorithm_resultfile:
                continue
            if (sub_level := algorithm_resultfile.replace(dataset,'').replace(f'_3DKHT.out','')) == '':
                continue
            result = Result.from_file(os.path.join(
                result_path, algorithm_resultfile))
            results_per_scene[dataset].append(result)
            results_per_param[int(sub_level.replace('_',''))].append(result)

    # calculate average results for each algorithm w.r.t scene type
    algo_results: List[Result] = []
    for param, scene in results_per_param.items():
        algorithm = f'3DKHT-'
        scene_p = scene_r = scene_f1 = 0.0
        scene_found = scene_all = 0
        total = per_plane = per_sample = 0.0
        for r in scene:
            scene_p += r.precision
            scene_r += r.recall
            scene_f1 += r.f1
            scene_found += r.detected
            scene_all += r.out_of
            total += r.time_total
            per_plane += r.time_per_plane
            per_sample += r.time_per_sample
        scene_p /= len(scene)
        scene_r /= len(scene)
        scene_f1 /= len(scene)
        total /= len(scene)
        per_plane /= len(scene)
        per_sample /= len(scene)
        param_average = Result(
            scene_p, scene_r, scene_f1, scene_found, scene_all, f'{param}_avg', algorithm, total, per_plane, per_sample)
        algo_results.append(param_average)
    # save results to file
    for result in algo_results:
        if 'results' not in os.listdir(rootFolder):
            os.mkdir(os.path.join(rootFolder, 'results'))
        filepath = os.path.join(
            root_folder, 'results', f'{result.algorithm}-{result.dataset}.out')
        result.to_file(filepath)


def get_df(results_folder: str):
    # load results
    results = [Result.from_file(os.path.join(results_folder, file))
               for file in os.listdir(results_folder) if '3DKHT' in file]
    results = [res for res in results if len(res.dataset) == 5]
    fig, axs = plt.subplots(1, 1)
    algo = '3DKHT'

    # filter results by algorithm
    algo_data = [res for res in results if algo in res.algorithm]
    algo_data.sort(key=lambda x: x.dataset.lower())
    # create algo dataframe
    df = pd.DataFrame(algo_data)
    precision = df.drop(columns=['detected','out_of','time_total','time_per_plane','time_per_sample'])
    precision.plot.bar(ax=axs)
    axs.set_xticklabels(f'{a}' for a,b in KHT_PARAMS)
    axs.set_xlabel('Octree Subdivision Level')
    axs.set_ylabel('Accuracy Metrics in %')
    # founds_df =  df.drop(columns=['precision','f1','recall', 'time_total','time_per_plane','time_per_sample'])
    # founds_df.plot.bar(ax=axs[1])
    # axs[1].set_xticklabels(f'{a}' for a,b in KHT_PARAMS)
    # axs[1].set_xlabel('subdivision_level')

    # times_df = df.drop(columns=['precision','f1','recall','detected', 'out_of','time_per_plane','time_per_sample' ])
    # times_df.plot.bar(ax=axs[2])

    # axs[2].set_xticklabels(f'{a}' for a,b in KHT_PARAMS)
    # axs[2].set_xlabel('subdivision_level')

    # algo_df = algo_df.rename(columns={'dataset': 'Scene Types'})
    # algo_df.plot.bar(x='Scene Types', ax=ax)  # , marker='o',label='rspd')

    # algo_data = [res for res in results if algo in res.algorithm]
    # algo_data.sort(key=lambda x: x.dataset)
    # df = pd.DataFrame(algo_data).drop(
    #     columns=['precision', 'recall', 'f1', 'detected', 'out_of'])
    # # sb.violinplot(data=df,ax=ax)
    # df.plot.bar(x='dataset', ax=ax)  # , marker='o',label='rspd')

    # axs[0].set_ylim(0.0, 1.0)
    plt.xticks(rotation=0)
    plt.suptitle('3DKHT')
    plt.show()


def kht_parameter_test(rootfolder: str, binaries_path: str):
    for dataset in os.listdir(rootfolder):
        dataset_path = os.path.join(rootfolder, dataset)
        # again, ignore files, results and datasets without GT
        if not os.path.isdir(dataset_path):
            continue
        if 'nope_' in dataset or dataset == 'results':
            continue
        algo = "3DKHT"
        binary = os.path.join(binaries_path, algo)
        cloud_file = os.path.join(
            dataset_path, f'{dataset}{ALGO_IN[algo]}')
        # create output folder if not already existing
        if algo not in os.listdir(dataset_path):
            os.mkdir(os.path.join(dataset_path, algo))
        else:
            for file in os.listdir(os.path.join(dataset_path, algo)):
                os.remove(os.path.join(dataset_path, algo, file))
        for s_level, d2p in KHT_PARAMS:
            result_file = os.path.join(
                dataset_path, algo, f'{dataset}-{s_level}')
            # run PDA on dataset
            print(f'Calling {algo} on {dataset}!')
            command = f'{binary} {cloud_file} {result_file} {s_level} {d2p}'
            os.system(command)


if __name__ == '__main__':
    fallback_root = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/Stanford3dDataset_v1.2_Aligned_Version/TEST"
    fallback_algo_binaries = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/AlgoBinaries"

    # input argument handling
    parser = argparse.ArgumentParser('BatchEvaluation')
    parser.add_argument('-R', '--root-folder', default=fallback_root,
                        help='Path to root directory which includes datasets to be tested')
    parser.add_argument('-A', '--algo-binaries',
                        default=fallback_algo_binaries)
    args = parser.parse_args()

    rootFolder = args.root_folder
    algorithm_binaries = args.algo_binaries

    # kht_parameter_test(rootFolder, algorithm_binaries)
    # kht_eval(rootFolder)
    kht_collect(rootFolder)
    get_df(os.path.join(rootFolder, 'results'))
