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
algos = ['RSPD','3DKHT', 'OPS', 'OBRG']
def tottime():
    data = dict()
    for algo in algos:
        data[algo] =  np.loadtxt(f'{algo}-data.txt', dtype = float)
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
    markers = ['o','v','x','+']
    for i, algo in enumerate(algos):
        d = data[algo]
        d = np.sort(d, axis=0)

        # d = np.array(d)
        if algo == '3DKHT':
            algo = "3D-KHT"
        
        if i > 1:
            plt.plot(d[:, 0], np.sum(d[:, 1:4], axis=1),marker=markers[i], label=algo, markersize=10)
        else:
            plt.plot(d[:, 0], np.sum(d[:, 1:3], axis=1),marker=markers[i], label=algo, markersize=10)
        ax = fig.gca()
        # ax.set_yscale('log')
        # ax.set_yticks([0.1, 1, 10,100,700]) #round(max(max_calc, max_pre)*1.1)])
        plt.yticks(fontsize=14)
        # ax.set_yticklabels(["$\\leq0.01$","1","10","100",'700'])
        print(ax.get_yticks())
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # ax.set_ylim(0.01,700)
        ax.set_xticks([0, 1_000_000, 3_000_000,5_000_000,7_000_000,9_000_000]) #round(max(max_calc, max_pre)*1.1)])
        ax.set_xticklabels([0,'$1.000.000$', '$3.000.000$','$5.000.000$','$7.000.000$', '$9.000.000$'])
        ax.grid(True)

        ax.tick_params(labelsize=20)
        
    fig.text(0.045, 0.5, 'Totale Berechnungszeit in Sekunden',
            ha='center', va='center', rotation='vertical',fontdict={'size':F_SIZE})
    fig.text(0.5, 0, 'Anzahl an Punkten', ha='center',
            va='bottom', rotation='horizontal',fontdict={'size':F_SIZE})

    # fig.axes[0].legend(loc='center', fancybox=True, shadow=True, ncol=5,prop={'size':24})
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # fig.axes[0].legend(loc='lower right', fancybox=True, shadow=True, ncol=5, prop={'size':F_SIZE})
    # plt.savefig("stanfordsvg.svg", format="svg")

    plt.show()

def pie():
    data = dict()
    for algo in algos:
        data[algo] =  np.loadtxt(f'{algo}-data.txt', dtype = float)
    fig = plt.figure(figsize=[100, 66])
    max_pre = -1
    max_calc = -1
    min_pre = 99
    min_calc = 99
    b = c = None
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
    markers = ['o','v','x','+']
    for i, algo in enumerate(algos):
        d = data[algo]
        d = np.sort(d, axis=0)

        # d = np.array(d)
        if algo == '3DKHT':
            algo = "3D-KHT"
        ax = fig.add_subplot(2,2, i+1)
        p = 0
        if i > 1:
            # p = ax.pie(np.sum(d,axis=0)[1:4],labels=['Pre-Processing','Plane Detection','Post-Processing'], autopct='%1.1f%%', startangle=-180)
            p = ax.pie(np.sum(d,axis=0)[1:4], autopct='%1.1f%%', startangle=-180)
        else:
            # p= ax.pie(np.sum(d,axis=0)[1:3],labels=['Pre-Processing','Plane Detection'], autopct='%1.1f%%', startangle=-180)
            p= ax.pie(np.sum(d,axis=0)[1:3],autopct='%1.1f%%', startangle=-180)
        ax.set_title(algo)
        # ax.set(ylabel='', title='', aspect='equal')
        # ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # lines_labels = [fig.axes[2].get_legend_handles_labels()]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # for line in lines:
    #     print(line.center, line.r, line.theta1, line.theta2)    
    lines = [
        matplotlib.patches.Wedge((0.0, 0.0), 1, -180.0, 8.097288608551025),
        matplotlib.patches.Wedge((0.0, 0.0), 1, 8.097288608551025, 165.42510151863098),
        matplotlib.patches.Wedge((0.0, 0.0), 1, 165.42510151863098, 180.00000670552254)]
    colors = [
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0), 
        (1.0, 0.4980392156862745, 0.054901960784313725, 1.0), 
        (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0)]
    for i, l in enumerate(lines):
        l._facecolor = colors[i]
    labels = ['Pre-Processing', 'Plane Detection', 'Post-Processing']
    fig.legend(lines, labels, loc='lower center', fancybox=True, shadow=True, ncol=5, prop={'size':F_SIZE})
    # fig.legend(lines, labels, loc='lower center', fancybox=True, shadow=True, ncol=5, prop={'size':F_SIZE})
    # fig.suptitle('Prozentuale Anteile der Berechnungsphasen')
    plt.show()

under_one = 0
def fin_tot(path):
    global under_one
    matplotlib.rcParams.update(
    {'font.size': 40, 'figure.subplot.bottom': 0.086, 'figure.subplot.top': 0.902, 'figure.subplot.right': 0.867, 'figure.subplot.left': 0.106})
    sizes = []
    for cfile in os.listdir(path):
        if not cfile.endswith('.txt'):
            continue
        # sizes.append([os.path.getsize(os.path.join(path,cfile))/1000000, int(cfile[6:10])])
        with open(os.path.join(path,cfile)) as file:
            sizes.append([len(file.readlines()), int(cfile[6:10])])
    fig = plt.figure(figsize=[140,66])
    sizes.sort(key= lambda x: x[1])
    sizes = np.array(sizes)
    print(sizes[-1])
    markers = ['o','v','x','+']
    ax = fig.add_subplot(1,1,1)
    ax.grid(True)
    l = ax.get_xlim()
    ax2 = ax.twinx()
    ax2.grid(True)
    
    A = {algo: {} for algo in algos}
    for i, algo in enumerate(algos):
        data = np.loadtxt(f'{algo}-{path.split("/")[-1]}.txt', dtype=float).tolist()
        # ax.set_title(algo, loc='left',y=0.9, x=0.01, pad=-16, fontdict={'size':F_SIZE})
        data.sort(key = lambda x : x[3])
        l = ax.get_xlim()
        
        data = np.array(data)
        frames = np.arange(len(data))
        frames2 = np.arange(len(sizes))
        for j, tf in enumerate(frames):
            A[algo][j] = data[j]
        if algo not in ['RSPD','3DKHT']:
            ax.plot(frames, np.sum(data[:,:3], axis=1),marker=markers[i],  label=algo)
        else:
            a = np.sum(data[:,:2], axis=1)
            under_one += len([x for x in a if x <= 1.0])
            ax.plot(frames, a,marker=markers[i],  label=algo)
        # ax.plot(0,0,'--',label="$size$")
        # ax = fig.gca()

        if i == 0:
            ax2.plot(frames2, sizes[:,0],'--',color='purple')
            # ax2.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            # ax2.set_ylim(0,1000_000)

            # ax2.set_yscale('log')
            # ax2.set_yticks([0, 1000, 10000, 100000,750000]) #round(max(max_calc, max_pre)*1.1)])
            # ax2.set_yticklabels(['0','$1.000$','$10.000$','$100.000$','$750.000$'])
    # ax2.set_yscale('log')
    # ax2.set_yticks([375, 3750,37500, 187500,375000,750_000]) #round(max(max_calc, max_pre)*1.1)])
    # ax2.set_yticklabels(['375','$3.750$','$37.500$','$187.500$','$375.000$','$750.000$'])
    # ax2.set_ylim(375,750000)
    # ax.set_yscale('log')
    # ax.set_yticks([0.1, 1, 10, 50, 100, 200 ]) #round(max(max_calc, max_pre)*1.1)])
    # ax.set_yticklabels(["0.1", "1", "10", "50",'100', '200' ])
    # ax.set_ylim(0.1,200)
    ax.set_xlim(-5,l[1]-5)

        # ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    fig.text(0.99, 0.5, 'Anzahl an Punkten', ha='center', va='center', rotation='vertical',fontdict={'size':F_SIZE})

    fig.text(0.045, 0.5, 'Totale Berechnungszeit in Sekunden',
            ha='center', va='center', rotation='vertical',fontdict={'size':F_SIZE})
    fig.text(0.5, 0, 'Individuelle Zeitschritte', ha='center',
            va='bottom', rotation='horizontal',fontdict={'size':F_SIZE})
    # fig.text(0.5,0, 'Individuelle Zeitschritte', ha='center', va='bottom', rotation='horizontal',fontdict={'size':F_SIZE})
    # box = fig.axes[0].get_position()
    # # fig.axes[0].set_position([box.x0, box.y0 + box.height * 0.1,
    # #                 box.width, box.height * 0.9])

    # # Put a legend below current axis
    # # fig.axes[0].legend(loc='center right', fancybox=True, shadow=True, ncol=5)
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper center', fancybox=True, shadow=True, ncol=5, prop={'size':F_SIZE})
    # plt.legend(labels=algos)
    plt.show()

    # maxlim = max([ax.get_ylim() for ax in fig.axes[::2]], key=lambda x: x[1])
    # minlim = min([ax.get_ylim() for ax in fig.axes[::2] if ax.get_ylim()[0] > 0],key=lambda x: x[0])
    # print(minlim, maxlim)    
    # for i, ax in enumerate(fig.axes):
    #     ax.tick_params(labelsize=20)

    #     # if i %2 == 1:
    #     if ax.get_title('left') in algos:
    #         ax.set_yscale('log')
    #         ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #         ax.set_yticks([0.01,1,10,round(maxlim[1])])
    #         ax.set_yticklabels([r'$\leq$0.01', "1","10",f'{round(maxlim[1])}'])
    #         ax.set_ylim(0.01,round(maxlim[1]))    
    #     else:
    #         ax.set_yticks([0, 1000, 10000, 100000,750000]) #round(max(max_calc, max_pre)*1.1)])
    #         ax.set_yticklabels(['0','$1.000$','$10.000$','$100.000$','$750.000$'])
    #         ax.set_yscale('log')
    #         ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

def fin_pie(path):
    matplotlib.rcParams.update(
    {'font.size': 40, 'figure.subplot.bottom': 0.052, 'figure.subplot.top': 0.926, 'figure.subplot.right': 0.955, 'figure.subplot.left': 0.086, 'figure.subplot.wspace':0, 'figure.subplot.hspace':0.143})
    data = dict()
    for algo in algos:
        data[algo] = np.loadtxt(f'{algo}-{path.split("/")[-1]}.txt', dtype=float)
    fig = plt.figure(figsize=[100, 66])
    # max_pre = -1
    # max_calc = -1
    # min_pre = 99
    # min_calc = 99
    # b = c = None
    # for da in data.values():
    #     d = np.array(da.copy())
    #     if (mp := max(d[:, 1])) > max_pre:
    #         max_pre = mp
    #     if (mp := max(d[:, 2])) > max_calc:
    #         max_calc = mp-180.0
    #     if (mp := min(d[:, 1])) < min_pre:
    #         min_pre = mp
    #     if (mp := min(d[:, 2])) < min_calc:
    #         min_calc = mp

    # print(f'{max_pre = }')
    # markers = ['o','v','x','+']
    for i, algo in enumerate(algos):
        d = data[algo]
        d = np.sort(d, axis=0)

        # d = np.array(d)
        if algo == '3DKHT':
            algo = "3D-KHT"
        ax = fig.add_subplot(2,2, i+1)
        p = 0
        if i > 1:
            a = np.sum(d,axis=0)[:3]
            # p = ax.pie([a[1],a[0],a[2]],labels=['Pre-Processing','Plane Detection','Post-Processing'], autopct='%1.1f%%', startangle=-180)
            p = ax.pie([a[1],a[0],a[2]], autopct='%1.1f%%', startangle=-180)
        else:
            a = np.sum(d,axis=0)[:2]
            # p= ax.pie([a[1],a[0]],labels=['Pre-Processing', 'Plane Detection'], autopct='%1.1f%%', startangle=-180)
            p= ax.pie([a[1],a[0]], autopct='%1.1f%%', startangle=-180)
        ax.set_title(algo)
        # ax.set(ylabel='', title='', aspect='equal')
        # ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # lines_labels = [fig.axes[2].get_legend_handles_labels()]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # for line in lines:
    #     print(line.center, line.r, line.theta1, line.theta2)
    lines = [
        matplotlib.patches.Wedge((0.0, 0.0), 1, -180.0, 142.44834423065186),
        matplotlib.patches.Wedge((0.0, 0.0), 1, 142.44834423065186, 179.3783739209175),
        matplotlib.patches.Wedge((0.0, 0.0), 1, 179.3783739209175, 179.99998826533556)]
    colors = [
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0), 
        (1.0, 0.4980392156862745, 0.054901960784313725, 1.0), 
        (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0)]
    for i, l in enumerate(lines):
        l._facecolor = colors[i]
    labels = ['Pre-Processing', 'Plane Detection', 'Post-Processing']
    fig.legend(lines, labels, loc='lower center', fancybox=True, shadow=True, ncol=5, prop={'size':F_SIZE})
    # fig.suptitle('Prozentuale Anteile der Berechnungsphasen')
    plt.show()
if __name__ == '__main__':
    datasets = ['auditorium','conferenceRoom','office','hallway']
    # for ds in datasets:
    # pie()
        # fin_tot(f'FIN-Dataset/{ds}')
    #     # plt.show()
    tottime()
    # pie()
    # print(under_one)