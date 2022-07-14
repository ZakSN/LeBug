import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

# draw a plot
def plot_results(layer_list, figdir):
    markers = ['>', '+', 'o', 'v', 'x', 'X', 'D', '|']
    colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    fig = plt.figure(figsize=(12, 3))
    plt.rc('axes', labelsize=15)
    plt.rc('axes', titlesize=15)
    gs = gridspec.GridSpec(1, 4)
    gs.update(wspace=0.1)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    first  = True
    for idx, configuration in enumerate(layer_list):
        results = configuration[2]
        layer_name = configuration[1]
        sampling_frequency = configuration[0]
        #ax = fig.add_subplot(1, 4, int(math.log(sampling_frequency, 2))+1, aspect=0.66)
        ax = plt.subplot(gs[int(math.log(sampling_frequency, 2))], aspect=0.66)
        for idx, key in enumerate(results[0]):
            plt.plot(*list(zip(*(results[0][key]))), label=key.replace('_', ' '), marker=markers[idx%8], color=colours[idx%6])
            plt.plot(*list(zip(*(results[1][key]))), marker=markers[idx%8], linestyle='dotted', color=colours[idx%6])
        plt.plot((2, 4, 8, 16), (2*(64/65), 4*(64/65), 8*(64/65), 16*(64/65)), linestyle='dotted', color='black', label="Ideal")
        plt.xscale('log', base=2)
        plt.yscale('log', base=2)
        plt.xlim(2, 8)
        plt.ylim(0.9, 8)
        plt.title("Sampling Period: " + str(sampling_frequency))
        #plt.xlabel("DELTA_SLOTS [# $\delta$s to compress]")
        plt.grid(visible=True)
        if sampling_frequency != 1:
            ax = plt.gca()
            ax.axes.yaxis.set_ticklabels([])
        if first:
            plt.figlegend(title='Firmware name', bbox_to_anchor = (0.58, -0.25),loc = 'lower center', ncol=4, borderaxespad=0, fontsize=12)
            #plt.figlegend(title='Firmware name', loc = (0.2, 0.25), ncol=len(results)+1)
            #plt.figlegend(title='Firmware name', loc = (0.88, 0.5))
            first = False
    #fig.suptitle("Compression Ratio vs. DELTA_SLOTS for Layer: " + layer_name, y=0.61)
    fig.supylabel("Compression Ratio", x=0.08)
    fig.supxlabel("D", x=0.21, y=-0.09)
    plt.savefig(os.path.join(figdir, layer_name.replace(' ', '_') + ".png"), bbox_inches='tight')
    #plt.show()

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

for layer_name in combined_results:
    plot_results(combined_results[layer_name], FIGURE_DIRECTORY)
