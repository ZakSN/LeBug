import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

# draw a plot
def plot_row(axes, data, row):
    markers = ['>', '+', 'o', 'v', 'x', 'X', 'D', '|']
    colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    for trial in data:
        sampling_period = trial[0]
        layer_name = trial[1]
        results = trial[2]
        col = int(math.log(sampling_period, 2))
        ax = axes[col]
        for idx, fw in enumerate(results[0]):
            ax.plot(*list(zip(*(results[0][fw]))), label=fw.replace('_', ' '), marker=markers[idx%8], color=colours[idx%6])
            ax.plot(*list(zip(*(results[1][fw]))), marker=markers[idx%8], linestyle='dotted', color=colours[idx%6])
        ax.plot((2, 4, 8, 16), (2*(64/65), 4*(64/65), 8*(64/65), 16*(64/65)), linestyle='dotted', color='black', label="Ideal")
        ax.set_aspect(0.66)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        ax.set_xlim(2, 8)
        ax.set_ylim(0.9, 8)
        if col != 0:
            ax.axes.yaxis.set_ticklabels([])
        if row == 0:
            ax.set_title("Sampling Period: " + str(sampling_period))
        if col == 0:
            ax.set_ylabel("Compression Ratio")
        if row == 2:
            ax.set_xlabel("D")
        if row == 2 and col == 3:
            plt.legend(title='firmware name', ncol=4, bbox_to_anchor = (0.6, -0.23), fontsize=12)
        ax.grid(visible=True)

RESULTS_DIRECTORY = 'results'
FIGURE_DIRECTORY = 'figures'
try:
    if not os.path.exists(FIGURE_DIRECTORY):
        os.makedirs(FIGURE_DIRECTORY)
except OSError:
    print('ERROR: could not make figure directory')
    exit()

def parse_results(rdir):
    result_files = os.listdir(rdir)
    results = {}
    for r in result_files:
        with open(os.path.join(rdir, r), 'rb') as file:
            data = pickle.load(file)
        sampling_frequency = int(r.split('_')[0])
        layer_name = ' '.join(r.split('.')[0].split('_')[1:])
        if layer_name not in results:
            results[layer_name] = []
        results[layer_name].append((sampling_frequency, layer_name, data))
    return results

results1 = parse_results("face_" + RESULTS_DIRECTORY)
results2 = parse_results("noface_" + RESULTS_DIRECTORY)

combined_results = {}

for key in results1:
    combined_results[key] = []
    for idx, item in enumerate(results1[key]):
        combined_results[key].append((item[0], item[1], (item[2], results2[key][idx][2])))

fig = plt.figure(figsize=(12, 8))
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=15)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
subfigs = fig.subfigures(nrows=3, ncols=1)
for layer_name in combined_results:
    def layer_name_lut(ln):
        if "6" in ln:
            return 0
        if "13" in ln:
            return 1
        if "20" in ln:
            return 2
    row = layer_name_lut(layer_name)
    subfig = subfigs[row]
    #subfig.suptitle(layer_name)
    axes = subfig.subplots(nrows=1, ncols=4)
    plot_row(axes, combined_results[layer_name], row)

plt.savefig(os.path.join(FIGURE_DIRECTORY, "compression_results.png"), bbox_inches='tight')
