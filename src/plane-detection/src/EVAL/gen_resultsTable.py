import os
from typing import Dict, List
from batchEvaluation import ALGOS
from classes import Result
# types = ['auditorium', 'conferenceRoom', 'copyRoom', 'hallway',
#  'lobby', 'lounge', 'office', 'openspace', 'pantry', 'storage', 'WC']
types = ['auditorium', 'conferenceRoom', 'hallway', 'office']


def get_dyn_res(rootpath,algo):
    res_per_algo: Dict[str, List[Result]] = {algo: [] for algo in ALGOS}
    for dataset in os.listdir(rootpath):
        if dataset == "TEST":
            continue
        for resfile in os.listdir(os.path.join(rootpath, dataset, 'results')):
            if not 'avg' in resfile:
                continue
            result = Result.from_file(os.path.join(
                rootpath, dataset, 'results', resfile))
            if result.algorithm != algo:
                continue
            print(f'{result.algorithm:10} {dataset:10}')
            print(result.precision)
            print(result.recall)
            print(result.f1)
            print(result.time_total)
            print(result.time_per_plane)
            print(result.time_per_sample)
            print('-------')
            res_per_algo[result.algorithm].append(result)
    for algo, results in res_per_algo.items():
        p = r = f1 = pre = calc = post = 0.0
        for result in results:
            p += result.precision
            r += result.recall
            f1 += result.f1
            pre += result.time_total
            calc += result.time_per_plane
            post += result.time_per_sample
        p /= len(results)
        r /= len(results)
        f1 /= len(results)
        pre /= len(results)
        calc /= len(results)
        post /= len(results)
        print(f'avgs for {algo}:')
        print(f'{p =}')
        print(f'{r =}')
        print(f'{f1 =}')
        print(f'{pre =}')
        print(f'{calc =}')
        print(f'{post =}')
        print('----')

def count_frames(root):
    ds_frames = {t:0 for t in types}
    for ds in types:
        _,_,files = list(os.walk(os.path.join(root,ds)))[0]
        if not isinstance(files, list):
            continue
        times = [int(x[6:10]) for x in files if x.endswith('.pcd')]
        ds_frames[ds] = max(times) - min(times)
    print(ds_frames)
def gen_latex(path, algo):
    p = rec = f1 = 0.0
    prec = calc = post = 0.0
    num = 0
    tot = o = 0
    pre = r'''\begin{table}[]
\centering
\begin{tabular}{c|ccccll}
                & Precision & Recall & F1-Score \\ \hline '''
    pre += '\n'
    print(f'{algo}')
    for t in types:
        found = outof = 0
        sceneresult = f'{algo}-{t}.out'
        r = Result.from_file(os.path.join(path, sceneresult))
        p += r.precision
        rec += r.recall
        f1 += r.f1
        prec += r.time_total
        calc += r.time_per_plane
        post += r.time_per_sample
        found += r.detected
        outof += r.out_of
        row = f'{t} & {round(r.precision, 4)}& {round(r.recall,4)}& {round(r.f1,4)}'
        if t != 'WC':
            row += '\t \\\ \n'
        else:
            row += '\t \\\ \\hline \n'
        pre += row
        print(f'\t{t:20} {found}/{outof}')
        tot += found
        o += outof
    X = 'Total'
    print(f'\t{X:20} {tot}/{o}')
    return
    p /= len(types)
    rec /= len(types)
    f1 /= len(types)
    pre += f'TOTAL \t &{round(p,4)} & {round(rec,4)} & {round(f1,4)} \n'
    pre += r'''\end{tabular}
\caption{}
\label{tab:my-table}
\end{table}'''
    print(pre)
    prec /= len(types)
    calc /= len(types)
    post /= len(types)
    print(f'{round(prec, 2)} & {round(calc,2)} & {round(post,5)}')
    print(f'{round(p*100,2)}\% & {round(rec*100,2)}\% & {round(f1*100,2)}\%')

# for algo in ALGOS:
#     gen_latex('Stanford3dDataset_v1.2_Aligned_Version/results', algo)
# get_dyn_res('FIN-Dataset', 'RSPD')
count_frames("FIN-Dataset")