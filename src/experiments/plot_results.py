import matplotlib.pyplot as plt
import numpy as np
import csv
import math

# inputs: axis to plot range on, dictionary of lists of x,y tuples with firmware
# names as keys
def plot_range(ax, to_plot, entropy_plot, range_id='', range_style='solid'):
    rng_cfg = {
        'raw' : ('tab:blue','>'),
        'distribution' : ('tab:orange','+'),
        'sparsity_count' : ('tab:green','o'),
        'spatial_sparsity' : ('tab:red','v'),
        'norm_check' : ('tab:purple','X'),
        'activation_predictiveness' : ('tab:brown','x'),
    }
    for fw in to_plot.keys():
        ax.plot(*list(zip(*to_plot[fw])),
                label=fw.replace('_', ' ') + range_id,
                marker=rng_cfg[fw][1],
                color=rng_cfg[fw][0],
                linestyle=range_style)
        ax.set_xscale('log', base=2)
        ax.set_xlim(2,8)
        if not entropy_plot:
            ax.set_aspect(0.66)
            ax.set_yscale('log', base=2)
            ax.set_ylim(0.9, 8)
        ax.grid(visible=True)

# return the csv file as a list of tuples
def read_results_file(file):
    raw = []
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            raw.append(row)
    return raw

# binary entropy formula, input is probability of one state
def entropy(p):
    return -p*math.log(p,2) - (1-p)*math.log(1-p,2)

def create_figure(entropy_plot):
    rows = 3
    columns = 4
    fig = plt.figure(figsize=(12,8))
    plt.rc('axes', labelsize=15)
    plt.rc('axes', titlesize=15)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    subfigs = fig.subfigures(nrows=rows, ncols=1)
    plt.subplots_adjust(left=0, bottom=0)
    fig.text(0.44445, -0.07, "D", ha='center', fontsize=20)
    if entropy_plot:
        major_vert = "Shannon Entropy"
    else:
        major_vert = "Compression Ratio"
    fig.text(-0.05, 0.5, major_vert, va='center', rotation='vertical', fontsize=20)
    axes = []
    for r in range(rows):
        axes.append(subfigs[r].subplots(nrows=1, ncols=columns))
    return axes
