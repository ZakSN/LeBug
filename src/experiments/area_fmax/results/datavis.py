import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patheffects as PathEffects
import numpy as np
from math import log

# matplot lib scaling is wonky, this script probably doesn't work right unless
# you've got exactly the same resolution screen as me...

fccm_21_data = np.array([
    # M = N/1,      N/4,           N/16,         N/N
    [[191,   9800], [177,   6200], [178,  5100], [178,  5100],], # N = 16
    [[160,  28700], [177,  14400], [173, 10900], [184, 10300],], # N = 32
    [[145,  95300], [169,  38100], [185, 23800], [177, 20300],], # N = 64
    [[129, 342600], [129, 114000], [165, 57000], [168, 40300],], # N = 128
])

lebug_HEAD_data = np.array([
    # M = N/1,         N/4,              N/16,            N/N
    [[149.19,   9522], [154.11,   5948], [153.33,  5031], [153.33,   5031],], # N = 16
    [[147.36,  28282], [144.8,   13977], [144.45, 10399], [148.26,   9812],], # N = 32
    [[126.69,  94511], [135.08,  37276], [145.29, 22859], [137.84,  19411],], # N = 64
    [[None, None], [130.33, 113258], [139.18, 55887], [141.18, 39341],], # N = 128
])

def parse_file(to_parse):
    def float_or_none(x):
        if x == 'FAIL':
            return None
        return float(x)
    data = []
    cont = True
    with open(to_parse) as log:
        for line in log:
            line = line.split()
            f = float_or_none(line[1])
            a = float_or_none(line[2])
            nmd = line[0].split("_")
            n = float_or_none(nmd[0])
            m = float_or_none(nmd[1])
            d = float_or_none(nmd[2])
            data.append((n,m,d,f,a))
    return data

#data = parse_file("202206092133_1b5d214_summary.rpt")
data = parse_file("202207081343_2c7529f_summary.rpt")
by_N = []
by_4 = []
by_16 = []
by_1 = []
for datum in data:
    if datum[1] == (datum[0]/datum[0]):
        by_N.append((datum[0],datum[2],(datum[3],datum[4])))
    if datum[1] == (datum[0]):
        by_1.append((datum[0],datum[2],(datum[3],datum[4])))
    if datum[1] == (datum[0]/4):
        by_4.append((datum[0],datum[2],(datum[3],datum[4])))
    if datum[1] == (datum[0]/16):
        by_16.append((datum[0],datum[2],(datum[3],datum[4])))

def plot_data_3d(r, l):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for dataset in [by_N, by_1, by_4, by_16]:
        to_plot = list(zip(*dataset))
        to_plot[2] = list(zip(*to_plot[2]))[r]
        x = np.array(to_plot[0])
        y = np.array(to_plot[1])
        z = np.array(to_plot[2])
        x = np.expand_dims(x, 1)
        y = np.expand_dims(y, 1)
        z = np.expand_dims(z, 1)
        ax.scatter(x, y, z,)
    ax.set_xlabel("N [# of vector elements]")
    ax.set_xticks([16, 32, 63, 128])
    ax.set_ylabel("DELTA_SLOTS [# of $\delta$s to compress]")
    ax.set_yticks([2, 4, 8])
    ax.set_zlabel(l)
    plt.show()

#plot_data_3d(0, "Frequency [MHz]")
#plot_data_3d(1, "Area [# of ALMs]")

def plot_data_2d(r, l, p, name=None, invert=False):
    def get_triple():
            x = int(log(pt[1], 2)) - 1
            y = int(log(pt[0]/16, 2))
            z = pt[2][r]
            return x, y, z
    def get_pdiff(y, idx, r, z, p):
        #f = fccm_21_data[y][idx][r]
        f = lebug_HEAD_data[y][idx][r]
        if f == None or z == None:
            return None
        pdiff = ((z - f)/f)*100
        if p == True:
            return pdiff
        else:
            return z
    fig = plt.figure(figsize=(11, 9))
    data = np.zeros((3, 4, 4))
    for idx, dataset in enumerate([(by_1, "M=N"), (by_4, "M=N/4"), (by_16, "M=N/16"), (by_N, "M=1")]):
        for pt in dataset[0]:
            x,y,z = get_triple()
            pdiff = get_pdiff(y, idx, r, z, p)
            data[x][y][idx] = pdiff
    min_z = np.nanmin(data)
    max_z = np.nanmax(data)
    for idx, dataset in enumerate([(by_1, "M=N"), (by_4, "M=N/4"), (by_16, "M=N/16"), (by_N, "M=1")]):
        ax = fig.add_subplot(2, 2, idx+1)
        for pt in dataset[0]:
            x,y,z = get_triple()
            pdiff = get_pdiff(y, idx, r, z, p)
            fs = 20
            if pdiff == None:
                txt = 'D.N.R.'
                fs=15
            else:
                txt = str("{:.0f}".format(pdiff))
                if p:
                    txt = txt + "%"
            plt.text(y, x, txt,
                     ha='center',
                     fontsize=fs,
                     path_effects=[PathEffects.withStroke(linewidth=3, foreground='w')])
        if invert == False:
            cmap = "Greys"
        else:
            cmap = "Greys_r"
        ax.imshow(data[:, :, idx], cmap=cmap, vmin=min_z, vmax=max_z)
        plt.title(dataset[1])
        plt.xlabel("N")
        ax.set_xticks(np.arange(0, 4, 1))
        ax.set_xticklabels([16, 32, 64, 128])
        ax.set_yticks(np.arange(0, 3, 1))
        ax.set_yticklabels([2, 4, 8])
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.ylabel("D")
    # uncomment for a supertitle -- takes too much space and is reproduced in
    # the caption
    #fig.suptitle(l, y=0.58)
    if name is not None:
        fig.savefig(name + ".png", dpi=300, bbox_inches='tight')
    plt.show()

plot_data_2d(0, "$F_{max}$ Percent Difference From LeBug-Head", True, name="fmax_percent_diff", invert=False)
plot_data_2d(1, "Area Percent Difference From LeBug-Head", True, name="area_percent_diff", invert=True)

#plot_data_2d(0, "Frequency [MHz], for: ", False)
#plot_data_2d(1, "Area [# ALM], for: ", False)
