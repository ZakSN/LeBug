import matplotlib.pyplot as plt
import numpy as np
import csv
import math

def plot_row(axes, data, row):
    rng_cfg = {
        'raw' : ('tab:blue','>'),
        'distribution' : ('tab:orange','+'),
        'sparsity_count' : ('tab:green','o'),
        'spatial_sparsity' : ('tab:red','v'),
        'norm_check' : ('tab:purple','X'),
        'activation_predictiveness' : ('tab:brown','x'),
    }
    for stride_ratio in data:
        l = list(data.keys())
        l.sort()
        col = l.index(stride_ratio)
        ax = axes[col]
        ranges = data[stride_ratio]
        for fw in ranges.keys():
            H_before = [(i[0], i[1]) for i in ranges[fw]]
            H_after = [(i[0], i[2]) for i in ranges[fw]]
            H_before.sort(key=lambda x: x[0])
            H_after.sort(key=lambda x: x[0])
            ax.plot(*list(zip(*H_before)), label=fw.replace('_', ' ') + ' I', marker=rng_cfg[fw][1], linestyle='dotted', color=rng_cfg[fw][0])
            ax.plot(*list(zip(*H_after)), label=fw.replace('_', ' ') + ' O', marker=rng_cfg[fw][1], color=rng_cfg[fw][0])
        #ax.plot((2, 4, 8, 16), (2*(64/65), 4*(64/65), 8*(64/65), 16*(64/65)), linestyle='dotted', color='black', label="Ideal")
        #ax.set_aspect(0.66)
        ax.set_xscale('log', base=2)
        #ax.set_yscale('log', base=2)
        ax.set_xlim(2, 8)
        #ax.set_ylim(0.9, 8)
        if col != 0:
            ax.axes.yaxis.set_ticklabels([])
        if row == 0:
            ax.set_title("Stride: "+str(stride_ratio))
        if row != 2:
            ax.axes.xaxis.set_ticklabels([])
        if row == 2 and col == 3:
            ax.legend(title='firmware name', ncol=4, bbox_to_anchor = (0.45, -0.23), fontsize=12)
        ax.grid(visible=True)

def parse_results(file):
    raw = []
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            raw.append(row)
    parsed = []
    for row in raw:
        parsed.append((row[0], float(row[1])/float(row[2]), row[4], float(row[3]), float(row[5]), float(row[6]), float(row[7])))
    return parsed

# apply the binary entropy formula assuming p = p(X=1)
def entropy(p):
    return -p*math.log(p,2) - (1-p)*math.log(1-p,2)

parsed = parse_results('compression_audio_results.csv')
layer_set = set()
stride_ratio_set = set()
data = {}
for p in parsed:
    if p[0] not in data:
        data[p[0]] = {}
    if p[1] not in data[p[0]]:
        data[p[0]][p[1]] = {}
    if p[2] not in data[p[0]][p[1]]:
        data[p[0]][p[1]][p[2]] = [(p[3], entropy(p[5]), entropy(p[6]))]
    else:
        data[p[0]][p[1]][p[2]].append((p[3], entropy(p[5]), entropy(p[6])))

fig = plt.figure(figsize=(12, 8))
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=15)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
subfigs = fig.subfigures(nrows=3, ncols=1)
plt.subplots_adjust(left=0, bottom=0)
for layer in data:
    def layer_lut(k):
        if 'conv' in k:
            return 0
        if 'act' in k:
            return 1
        if 'batch' in k:
            return 2
    row = layer_lut(layer)
    subfigs[row].supylabel(layer.replace('_', ' '), x=0.91, fontsize=15)
    axes = subfigs[row].subplots(nrows=1, ncols=4)
    plot_row(axes, data[layer], row)
fig.text(0.44445, -0.07, "D", ha='center', fontsize=20)
fig.text(-0.05, 0.5, "Shannon Entropy", va='center', rotation='vertical', fontsize=20)
plt.savefig('entropy_results.png', bbox_inches='tight')
