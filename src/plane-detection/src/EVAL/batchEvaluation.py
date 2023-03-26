from logging import root
import argparse
import matplotlib.pyplot as plt
import os
from typing import Dict, List
from classes import Result
from evaluate import evaluate
import pandas as pd
import numpy as np
from fileio import create_pcd
import seaborn as sb
import sys
from tqdm import tqdm
sys.path.append('/home/pedda/Documents/coding/OBRG/')
sys.path.append('/home/pedda/Documents/coding/RSPyD/')
import rspyd
import obrg
from time import time
import open3d as o3d
plt.rcParams.update({'font.size': 19})

# globals
ALGOS = ['RSPD', 'OPS', '3DKHT', 'OBRG', 'RSPyD']
ALGO_ext = {'RSPD': '.geo', 'OPS': '', '3DKHT': '','RSPyD': '', 'OBRG': '', 'Application':''}
ALGO_IN = {'RSPD': '.txt','RSPyD': '.txt', 'OPS': '.pcd', '3DKHT': '.txt', 'OBRG': '.txt', 'Application':'.txt'}


def get_file_sizes_time_pairs(root_folder: str):
    results_algo_scenetypes: Dict[str, Dict[str, List]] = {
        algo: dict() for algo in ALGOS}
    sizes = []
    for i in range(1, 7):
        area = os.path.join(root_folder, f'Area_{i}')
        for dataset in os.listdir(area):
            if 'nope_' in dataset or 'result' in dataset or 'DS' in dataset:
                continue
            dataset_path = os.path.join(area, dataset)
            if 'results' not in os.listdir(dataset_path):
                continue
            result_path = os.path.join(dataset_path, 'results')
            size = os.path.getsize(os.path.join(dataset_path, dataset+'.txt'))
            sizes.append((size,f'{area}/{dataset}'))
            for algo_result in os.listdir(result_path):
                result = Result.from_file(
                    os.path.join(result_path, algo_result))
                dataset = result.dataset.split('_')[0]
                if dataset not in results_algo_scenetypes[result.algorithm].keys():
                    results_algo_scenetypes[result.algorithm][dataset] = []
                results_algo_scenetypes[result.algorithm][dataset].append(
                    [size, result.time_per_plane])
    # print(sum(sizes)/len(sizes)/1000000)
    print(f'{min(sizes,key=lambda x: x[0])[0]/1000000}')
    print(f'{min(sizes,key=lambda x: x[0])[1]}')
    print(f'{max(sizes,key=lambda x: x[0])[0]/1000000}')
    print(f'{max(sizes,key=lambda x: x[0])[1]}')

    return
    print(1)
    # fig = plt.figure(figsize=[20, 15])
    # # for i, algo in enumerate(ALGOS):
    # ax = fig.add_subplot(111)
    # ax.set_title('RSPD')
    # colors = np.random.rand(len(results_algo_scenetypes['RSPD'].keys()), 3)
    # for i, (scenetype, size_time) in enumerate(results_algo_scenetypes['RSPD'].items()):
    #     if scenetype == 'auditorium':
    #         continue
    #     times = sizes = []
    #     for s_t in size_time:
    #         size, time = s_t
    #         times.append(time)
    #         sizes.append(size/1000000)
    #     plt.scatter(times, sizes, label=scenetype)
    # ax.legend()
    # plt.show()

    fig = plt.figure(figsize=[20, 15])
    for i, algo in enumerate(ALGOS):
        ax = fig.add_subplot(1,len(ALGOS), i+1)
        ax.set_title(algo)
        alltimes=  []
        allsizes = []
        times = []
        scenetypes = sorted(list(results_algo_scenetypes[algo].keys()), key=lambda x : x.lower())
        sizes = []
        for i, st in enumerate(scenetypes):
            times.append([])
            sizes.append([])
            for s_t in results_algo_scenetypes[algo][st]:
                size, time = s_t
                times[i].append(time)
                alltimes.append(time)
                allsizes.append(size/1000000)
                sizes[i].append(size/1000000)
        ax.violinplot(times, showmedians=True)
        print(f'avg time for {algo}: {sum(alltimes)/len(alltimes)}')
        print(f'avg size: {sum(allsizes)/len(allsizes)}')
        if algo == "RSPD":
            ax.set_ylabel("time in s",fontsize=22)
        ax.set_xticks([i+1 for i in range(len(scenetypes))])
        ax.set_xticklabels(scenetypes)
        fig.autofmt_xdate(rotation=45)
    # for i, algo in enumerate(ALGOS):
    #     ax = fig.add_subplot(2, len(ALGOS), len(ALGOS) +i+1)
    #     ax.set_title(algo)
    #     times = []
    #     scenetypes = sorted(list(results_algo_scenetypes[algo].keys()), key=lambda x : x.lower())
    #     sizes = []
    #     for i, st in enumerate(scenetypes):
    #         times.append([])
    #         sizes.append([])
    #         if 'audi' in st:
    #             times[i].append(-1)
    #             sizes[i].append(-1)
    #             continue
    #         for s_t in results_algo_scenetypes[algo][st]:
    #             size, time = s_t
    #             times[i].append(time)
    #             sizes[i].append(size/1000000)
        # ax.violinplot(sizes, showmedians=True)
        # if algo == "RSPD":
        #     ax.set_ylabel("size in mb",fontsize=22)
        # ax.set_xticks([i+1 for i in range(len(scenetypes))])
        # ax.set_xticklabels(scenetypes)
        # fig.autofmt_xdate(rotation=45)
    plt.show()


def vis_total_results(root_path: str, algos=ALGOS):
    results_folder = os.path.join(root_path, 'results')
    results = [Result.from_file(os.path.join(results_folder, file))
               for file in os.listdir(results_folder) if file.endswith('.out') and not 'avg' in file]
    max_time = max(results, key=lambda x: x.time_total)
    fig = plt.figure(figsize=[20, 15])
    for i, algo in enumerate(algos):
        ax = fig.add_subplot(1, len(algos), i+1)
        ax.set_title(algo)

        # # filter results by algorithm
        algo_data = [res for res in results if res.algorithm == algo]
        if len(algo_data) == 0:
            continue
        algo_data.sort(key=lambda x: x.dataset.lower())
        # create algo dataframe
        algo_df = pd.DataFrame(algo_data).drop(
            columns=['detected', 'out_of', 'time_total', 'time_per_plane', 'time_per_sample'])
        algo_df = algo_df.rename(columns={'dataset': 'Scene Types'})
        algo_df.plot.bar(x='Scene Types', ax=ax)  # , marker='o',label='rspd')
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("")
        ax.get_xaxis().set_label("")
        ax.legend().remove()
    fig.autofmt_xdate(rotation=45)
    fig.supxlabel('Scene Types')
    plt.show()

    fig = plt.figure(figsize=[20, 15])
    for i, algo in enumerate(algos):
        ax = fig.add_subplot(1, len(algos), i+1)
        ax.set_title(algo)

        algo_data = [res for res in results if res.algorithm == algo]
        if len(algo_data) == 0:
            continue
        algo_data.sort(key=lambda x: x.dataset.lower())
        df = pd.DataFrame(algo_data).drop(
            columns=['precision', 'recall', 'f1', 'detected', 'out_of', 'time_total', 'time_per_sample'])
        # sb.violinplot(data=df,ax=ax)
        # df.plot.bar(x='dataset', ax=ax)  # , marker='o',label='rspd')

        ax.set_xlabel("")
        ax.get_xaxis().set_label("")
        ax.legend().remove()
    fig.autofmt_xdate(rotation=45)
    fig.supxlabel('Scene Type')
    plt.show()


def combine_area_results(root_path: str, algos=ALGOS):
    # gather results per area
    results_per_area: Dict[str, List[Result]] = {
        f'Area_{i}': [] for i in range(1, 7)}
    if 'results' not in os.listdir(root_path):
        os.mkdir(os.path.join(root_path, 'results'))
    for i in range(1, 7):
        area = f'Area_{i}'
        results = os.path.join(root_path, area, 'results')
        for res in os.listdir(results):
            if 'avg' in res:
                continue
            r = Result.from_file(os.path.join(results, res))
            results_per_area[area].append(r)
    # gather algorithm results
    results_algos_scenetypes: Dict[str, Dict[str, List[Result]]] = {
        algo: dict() for algo in algos}
    for area, results in results_per_area.items():
        for result in results:
            if result.dataset not in results_algos_scenetypes[result.algorithm].keys():
                results_algos_scenetypes[result.algorithm][result.dataset] = []
            results_algos_scenetypes[result.algorithm][result.dataset].append(
                result)
    # average results and write to file
    for algo, sceneresults in results_algos_scenetypes.items():
        for scenetype, results in sceneresults.items():
            p = r = f1 = 0.0
            found = outof = 0
            pre = time = post = 0.0
            
            for result in results:
                p += result.precision
                r += result.recall
                f1 += result.f1
                found += result.detected
                outof += result.out_of
                pre += result.time_total
                time += result.time_per_plane
                post += result.time_per_sample
            p /= len(results)
            r /= len(results)
            f1 /= len(results)
            time /= len(results)
            pre /= len(results)
            post /= len(results)
            avg = Result(p, r, f1, found, outof, scenetype, algo, pre,time,post)
            Result.to_file(avg, os.path.join(
                root_path, 'results', f'{algo}-{scenetype}.out'))


def get_df(results_folder: str, algos=ALGOS):
    # load results
    results = [Result.from_file(os.path.join(results_folder, file))
               for file in os.listdir(results_folder) if file.endswith('.out') and not 'avg' in file]
    fig, axs = plt.subplots(1, len(algos))
    fig.set_size_inches(20, 15)
    for ax, algo in zip(axs, algos):
        ax.set_title(algo)

        # filter results by algorithm
        algo_data = [res for res in results if res.algorithm == algo]
        if len(algo_data) == 0:
            continue
        algo_data.sort(key=lambda x: x.dataset.lower())
        # create algo dataframe
        algo_df = pd.DataFrame(algo_data).drop(
            columns=['detected', 'out_of', 'time_total', 'time_per_plane', 'time_per_sample'])
        algo_df = algo_df.rename(columns={'dataset': 'Scene Types'})
        algo_df.plot.bar(x='Scene Types', ax=ax)  # , marker='o',label='rspd')
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("")
        ax.get_xaxis().set_label("")
        ax.legend().remove()

    fig.autofmt_xdate(rotation=45)
    fig.supxlabel('Scene Types')
    area = results_folder.rsplit('/', 2)[1].lower()
    fname = f'{area}_acc.png'

    # plt.savefig(os.path.join('/home/pedda/Documents/uni/BA/Thesis/Document/images',fname))
    plt.show()
    # plt.close()
    fig, axs = plt.subplots(1, len(algos))
    fig.set_size_inches(20, 15)

    for ax, algo in zip(axs, algos):
        ax.set_title(algo)

        algo_data = [res for res in results if res.algorithm == algo]
        if len(algo_data) == 0:
            continue
        algo_data.sort(key=lambda x: x.dataset.lower())
        df = pd.DataFrame(algo_data).drop(
            columns=['precision', 'recall', 'f1', 'detected', 'out_of', 'time_total', 'time_per_sample'])
        # sb.violinplot(data=df,ax=ax)
        df.plot.bar(x='dataset', ax=ax)  # , marker='o',label='rspd')
        ax.set_xlabel("")
        ax.get_xaxis().set_label("")
        ax.legend().remove()
    fig.autofmt_xdate(rotation=45)
    fig.supxlabel('Scene Type')
    fname = f'{area}_time.png'
    plt.show()
    # TODO save fig to /$root_folder/figures


def collect_results(root_folder: str, algos=ALGOS):
    # scene -> [Result_RSPD, Result_OPS, ...]
    results_per_scene: Dict[str, List[Result]] = dict()
    # algo -> [scene_type -> [scene_specific_result]]
    results_per_algo: Dict[str, Dict[str, List[Result]]] = {
        algo: dict() for algo in algos}
    datasets = os.listdir(root_folder)
    for dataset in datasets:
        if os.path.isfile(os.path.join(root_folder, dataset)):
            # skip files
            continue
        if dataset.startswith('nope_') or dataset == 'results':
            # skip non-datasets (without GT or results directory)
            continue
        results_per_scene[dataset] = []
        result_path = os.path.join(root_folder, dataset, 'results')
        scene_type = dataset.split('_')[0]

        # populate result dicts, per_scene is currently unused, might be useful later
        for algorithm_resultfile in os.listdir(result_path):

            result = Result.from_file(os.path.join(
                result_path, algorithm_resultfile))
            results_per_scene[dataset].append(result)
            if result.algorithm not in algos:
                continue
            if scene_type not in results_per_algo[result.algorithm].keys():
                results_per_algo[result.algorithm][scene_type] = []
            results_per_algo[result.algorithm][scene_type].append(result)

    # calculate average results for each algorithm w.r.t scene type
    algo_results: List[Result] = []
    for algorithm, scenes_results in results_per_algo.items():
        for scene_type, scene in scenes_results.items():
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
            scene_type_average = Result(
                scene_p, scene_r, scene_f1, scene_found, scene_all, f'{scene_type}', algorithm, total, per_plane, per_sample)
            algo_results.append(scene_type_average)
    # save results to file
    for result in algo_results:
        if 'results' not in os.listdir(rootFolder):
            os.mkdir(os.path.join(rootFolder, 'results'))
        filepath = os.path.join(
            root_folder, 'results', f'{result.algorithm}-{result.dataset}.out')
        result.to_file(filepath)


def batch_evaluate(root_folder: str, algos=ALGOS):
    datasets = os.listdir(root_folder)
    for dataset in datasets:
        # ignore files, results and datasets without GT
        if os.path.isfile(os.path.join(root_folder, dataset)):
            continue
        if dataset.startswith('nope_') or dataset.startswith('results'):
            continue
        dataset_path = os.path.join(root_folder, dataset)
        gt_path = os.path.join(dataset_path, "GT")
        methods = []
        cloud_filename = dataset + '.txt'
        cloud_path = os.path.join(dataset_path, cloud_filename)
        for algo in algos:
            if algo in os.listdir(dataset_path):
                methods.append(os.path.join(dataset_path, algo))
        for algo_path in methods:
            evaluate(cloud_path, gt_path, algo_path)


def batch_detect(rootfolder: str, binaries_path: str, algos=ALGOS) -> None:
    for dataset in os.listdir(rootfolder):
        dataset_path = os.path.join(rootfolder, dataset)
        # again, ignore files, results and datasets without GT
        if not os.path.isdir(dataset_path):
            continue
        if 'nope_' in dataset or dataset == 'results':
            continue
        run_specific(dataset_path, binaries_path, algos)

def run_specific(dataset_path, binaries_path, algos=ALGOS):
    dataset = dataset_path.rsplit('/',1)[-1]
    for algo in algos:
        # if algo in os.listdir(dataset_path):
        #     continue
        # get input params for given algorithm
        binary = os.path.join(binaries_path, algo)
        cloud_file = os.path.join(
            dataset_path, f'{dataset}{ALGO_IN[algo]}')
        result_file = os.path.join(
            dataset_path, algo, f'{dataset}{ALGO_ext[algo]}')
        # create pcd file if needed (OPS)
        if cloud_file not in os.listdir(dataset_path):
            create_pcd(cloud_file.replace('.pcd', '.txt'))
        # create output folder if not already existing
        if algo not in os.listdir(dataset_path):
            os.mkdir(os.path.join(dataset_path, algo))
        else:
            for file in os.listdir(os.path.join(dataset_path, algo)):
                os.remove(os.path.join(dataset_path, algo, file))
        # run PDA on dataset
        if algo == 'OBRG':
            obrg.calculate(cloud_file, dataset_path)
        elif algo == 'RSPyD':
            points = np.loadtxt(cloud_file, usecols=(0, 1, 2), delimiter=' ')
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud = point_cloud.voxel_down_sample(voxel_size=0.05)
            point_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            t = time()
            kd_tree = o3d.geometry.KDTreeFlann(point_cloud)
            connectivity = rspyd.ConGraph(len(point_cloud.points))
            for i in tqdm(range(len(point_cloud.points))):
                if len(neighbors := connectivity.get_neighbors(i)) >= 30:
                    connectivity.add_node(i, neighbors)
                else:
                    [_, neighbors, _] = kd_tree.search_knn_vector_3d(
                        point_cloud.points[i], 31)
                    connectivity.add_node(i, neighbors[1:])

            detector = rspyd.Detector(point_cloud, connectivity)
            detector.config(0.5, 0.258819, 0.75)
            t1 = time()
            pre = t1-t
            planes = detector.detect()
            t2 = time()-t1
            np.savetxt(os.path.join(dataset_path, algo,f'{dataset}-times.txt'),[pre,t2, -1], delimiter=' ')
            for i, p in enumerate(planes):
                a = np.take(np.asarray(point_cloud.points), p.inliers, axis=0)
                np.savetxt(os.path.join(dataset_path, algo,f'plane-{i}.txt'), a, delimiter=' ')
        else:
            print(f'Calling {algo} on {dataset}!')
            command = f'{binary} {cloud_file} {result_file}'
            os.system(command)

def get_df2(results_folder: str, algos=ALGOS):
    # load results
    results = [Result.from_file(os.path.join(results_folder, file))
               for file in os.listdir(results_folder) if file.endswith('.out') and not 'avg' in file]
    fig, axs = plt.subplots(1, len(algos))
    fig.set_size_inches(20, 15)
    i = 0
    for ax, algo in zip(axs, algos):
        ax.set_title(algo)
        ax.set_xlim(0.0,1.0)
        # filter results by algorithm
        algo_data = [res for res in results if res.algorithm == algo]
        if len(algo_data) == 0:
            continue
        algo_data.sort(key=lambda x: x.dataset.lower(), reverse=True)
        # create algo dataframe
        algo_df = pd.DataFrame(algo_data).drop(
            columns=['detected', 'out_of', 'time_total', 'time_per_plane', 'time_per_sample'])
        algo_df = algo_df.rename(columns={'dataset': 'Scene Types'})
        # algo_df.plot.barh(x='Scene Types',y=['precision','recall','f1'], ax=ax)  # , marker='o',label='rspd')
        algo_df.plot.barh(ax=ax)
        if i == 0:
            ax.set_yticklabels(algo_df['Scene Types'])
            i += 1
        else:
            ax.set_yticklabels([])
        ax.set_xticks([0,0.25,0.5,0.75,1.0])
        ax.set_xticklabels([0,25,50,75,100])
        ax.legend().remove()
    # lines_labels = [axs.flat[0].get_legend_handles_labels()]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # fig.legend(lines, labels)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.suptitle('Accuracy Metrics in %',va='bottom', y=0.05)
    plt.show()


if __name__ == '__main__':
    fallback_root = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/Stanford3dDataset_v1.2_Aligned_Version/"
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
    # run_specific("/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/Stanford3dDataset_v1.2_Aligned_Version/Area_5/office_3",algorithm_binaries, ['OPS'])
    # for i in range(2,3):
    # batch_detect(os.path.join(rootFolder, 'Area_3'), algorithm_binaries, ['RSPyD'])
    # batch_evaluate(os.path.join(rootFolder, f'Area_{3}'), ['RSPyD'])
    # collect_results(os.path.join(rootFolder, f'Area_{3}'), ['RSPyD'])
    combine_area_results('Stanford3dDataset_v1.2_Aligned_Version')
    get_df(os.path.join(rootFolder,"results"))
    # get_df2(os.path.join(rootFolder,"results"))
    # vis_total_results('Stanford3dDataset_v1.2_Aligned_Version')
    # get_file_sizes_time_pairs(rootFolder)
