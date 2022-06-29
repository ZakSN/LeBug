import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

# draw a plot
def plot_results(layer_list, figdir):
    markers = ['>', '+', 'o', 'v', 'x', 'X', 'D', '|']
    fig = plt.figure(figsize=(20, 20))
    first  = True
    for idx, configuration in enumerate(layer_list):
        results = configuration[2]
        layer_name = configuration[1]
        sampling_frequency = configuration[0]
        ax = fig.add_subplot(1, 4, int(math.log(sampling_frequency, 2))+1, aspect=0.66)
        for idx, key in enumerate(results):
            plt.plot(*list(zip(*results[key])), label=key.replace('_', ' '), marker=markers[idx%8])
        plt.plot((2, 4, 8, 16), (2, 4, 8, 16), linestyle='dotted', color='black', label="Ideal")
        plt.xscale('log', base=2)
        plt.yscale('log', base=2)
        plt.xlim(2, 8)
        plt.ylim(0.9, 8)
        plt.title("Sampling Period: " + str(sampling_frequency))
        plt.xlabel("DELTA_SLOTS [# $\delta$s to compress]")
        plt.grid(visible=True)
        if int(math.log(sampling_frequency, 2)) == 0:
            plt.ylabel("Compression Ratio [(# vectors)/(# TB addrs)]")
        if first:
            plt.figlegend(title='Firmware name', bbox_to_anchor = (0.51, 0.355),loc = 'lower center', ncol=len(results)+1, borderaxespad=0)
            #plt.figlegend(title='Firmware name', loc = (0.2, 0.25), ncol=len(results)+1)
            #plt.figlegend(title='Firmware name', loc = (0.88, 0.5))
            first = False
    fig.suptitle("Compression Ratio vs. DELTA_SLOTS for Layer: " + layer_name, y=0.61)
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

result_files = os.listdir(RESULTS_DIRECTORY)
results = {}
for r in result_files:
    with open(os.path.join(RESULTS_DIRECTORY, r), 'rb') as file:
        data = pickle.load(file)
        sampling_frequency = int(r.split('_')[0])
        layer_name = ' '.join(r.split('.')[0].split('_')[1:])
        if layer_name not in results:
            results[layer_name] = []
        results[layer_name].append((sampling_frequency, layer_name, data))

for layer_name in results:
    plot_results(results[layer_name], FIGURE_DIRECTORY)
