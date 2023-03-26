import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from classes import Result
matplotlib.rcParams.update(
    {'font.size': 40, 'figure.subplot.bottom': 0.06, 'figure.subplot.top': 0.95})
root = 'Stanford3dDataset_v1.2_Aligned_Version'
F_SIZE = 40
# algo -> [[size, pre, calc, post]]
data = {'RSPD': [],
        'RSPyD':[],
        '3DKHT': [],
        'OPS': [],
        'OBRG': []
        }
st = [0, 0]
k = 0
for i in range(3, 4):
    area = os.path.join(root, f'Area_{i}')
    for scene in os.listdir(area):
        if scene == 'results' or 'nope' in scene or 'DS' in scene:
            continue
        cloudfile = f'{scene}.txt'
        size = os.path.getsize(os.path.join(area, scene, cloudfile))
        # if (size/1000000) > 39:
        #     continue
        points = None
        with open(os.path.join(area, scene, cloudfile)) as file:
            points = len(file.readlines())
            st[0] += points
            st[1] += (size/1000000)
        k += 1
        # if size/1000000 > 250:
        #     continue
        for algo in data.keys():
            result = Result.from_file(os.path.join(
                area, scene, 'results', f'{scene}_{algo}.out'))
            if result.algorithm =='OBRG' and result.time_per_plane > 1000:
                continue
            data[algo].append(
                [points, result.time_total, result.time_per_plane, result.time_per_sample, result.precision, result.recall, result.f1])
print(f'{st = }')
print(f'{round(st[0]/k)/round(st[1]/k)}')
# exit()
fig = plt.figure(figsize=[100, 66])
max_pre = -1
max_calc = -1
min_pre = 99
min_calc = 99
for da in data.values():
    d = np.array(da.copy())
    if (mp := max(d[:, 1])) > max_pre:
        max_pre = mp
    if (mp := max(d[:, 2])) > max_calc:
        max_calc = mp
    if (mp := min(d[:, 1])) < min_pre:
        min_pre = mp
    if (mp := min(d[:, 2])) < min_calc:
        min_calc = mp

print(f'{max_pre = }')
for i, algo in enumerate(data.keys()):
    d = data[algo].copy()
    d.sort(key=lambda x: x[0])
    d = np.array(d)
    print(f'{algo:10}:{(sum(d[:,0])/len(d))/1000000}')
    print(f'{algo:10}:{sum(d[:,1])/len(d)}')
    print(f'{algo:10}:{sum(d[:,2])/len(d)}')
    print(f'{algo:10}:{sum(d[:,3])/len(d)}')
    print(f'{algo:10}:{sum(d[:,4])/len(d)}')
    print(f'{algo:10}:{sum(d[:,5])/len(d)}')
    print(f'{algo:10}:{sum(d[:,6])/len(d)}')
    # d = d * [1/1000000, 1, 1, 1, 1, 1, 1]
    ax = fig.add_subplot(3,2, i+1)
    if algo == '3DKHT':
        algo = "3D-KHT"
    ax.set_title(algo, loc='left',y=0.93, x=0.01, pad=-16, fontdict={'size':F_SIZE})
    
    # ax.set_title(algo, fontdict={'size':F_SIZE})
    # ax.set_yscale('log')
    # ax2 = ax.twinx()
    # ax.set_xscale('log')
    # ax.set_ylim(min(min_pre, min_calc), max(max_calc, max_pre)*1.1)
    # ax2.set_ylim(min(min_pre, min_calc), max(max_calc, max_pre)*1.1)
    # ax.scatter(d[:, 0], d[:, 1], s=14)  # label='$t_{pre}$')
    # ax.scatter(d[:, 0], d[:, 2], s=14)  # label='$t_{calc}$')
    # ax.plot(d[:, 0], d[:, 1],linewidth=2, marker='.', label='$t_{pre}$')
    # ax.plot(d[:, 0], d[:, 2],linewidth=2, marker='.', label='$t_{calc}$')
    # d = np.array(d)
    if algo == '3DKHT':
        algo = "3D-KHT"
    # ax = fig.add_subplot(2,2, i+1)
    p = 0
    if i > 2:
        # p = ax.pie(np.sum(d,axis=0)[1:4],labels=['Pre-Processing','Plane Detection','Post-Processing'], autopct='%1.1f%%', startangle=-180)
        p = ax.pie(np.sum(d,axis=0)[1:4], autopct='%1.1f%%', startangle=-180)
    else:
        # p= ax.pie(np.sum(d,axis=0)[1:3],labels=['Pre-Processing','Plane Detection'], autopct='%1.1f%%', startangle=-180)
        p= ax.pie(np.sum(d,axis=0)[1:3],autopct='%1.1f%%', startangle=-180)
    ax.set_title(algo)

    # if i > 1:
    #     # ax.plot(d[:, 0], d[:, 3], marker='.', label='$t_{post}$')
    #     ax.plot(d[:, 0], np.sum(d[:, 1:4], axis=1),
    #             marker='.',  label='$t_{tot}$')
    # else:
    #     # ax.plot(-1, -1, marker='.',  label='$t_{post}$')
    #     ax.plot(d[:, 0], np.sum(d[:, 1:3], axis=1),
    #             marker='.',  label='$t_{tot}$')

    # ax.set_yticks([0.0, 0.01,0.1,1,10,100])
    # ax.set_yticklabels([0, 0.1, 1, 10, 100, 200])
    # ax2.set_yticklabels([0, 0.1, 1, 10, 100, 200])
    # ticks =[float(format(x, 'f')) for x in ax.get_yticks()]
    # print(ticks)
    # ax.set_yticks(ticks)
    # ax.set_yticks([0, 10, 100, 200,500, 1000])
    # ax.set_yscale('log')
    # ax.set_yticks([0.01, 1, 10,100,700]) #round(max(max_calc, max_pre)*1.1)])
    # plt.yticks(fontsize=14)
    # ax.set_yticklabels(["$\\leq0.01$","1","10","100",'700'])
    # print(ax.get_yticks())
    # ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # # ax.set_ylim(min(min_pre, min_calc),700)
    # ax.set_ylim(0.01,700)
    # # ax.set_xticks([0, 1_000_000, 3_000_000,5_000_000,7_000_000,9_000_000]) #round(max(max_calc, max_pre)*1.1)])
    # # ax.set_xticklabels([0, '$1.000.000$', '$3.000.000$','$5.000.000$','$7.000.000$', '$9.000.000$'])
    # ax.grid(True)

    # ax.ticklabel_format(axis='x',style='plain')
    # if i <  2:
    #     ax.set_xticklabels([])
        # ax.get_xaxis().set_visible(False)
    # ax.tick_params(labelsize=20)
    
    # ax.set_xscale('log')
fig.text(0.045, 0.5, '$t_{pre}, t_{calc}, t_{post}, t_{tot}$ in Sekunden',
         ha='center', va='center', rotation='vertical',fontdict={'size':F_SIZE})
fig.text(0.5, 0, 'Anzahl an Punkten', ha='center',
         va='bottom', rotation='horizontal',fontdict={'size':F_SIZE})

# fig.axes[0].legend(loc='center', fancybox=True, shadow=True, ncol=5,prop={'size':24})
lines_labels = [fig.axes[0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='center', fancybox=True, shadow=True, ncol=5, prop={'size':F_SIZE})


# plt.savefig("stanfordsvg.svg", format="svg")

plt.show()
