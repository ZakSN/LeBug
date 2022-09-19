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
        'ideal' : ('black', '.')
    }
    for fw in to_plot.keys():
        if fw == 'ideal':
            rs='dotted'
        else:
            rs=range_style
        ax.plot(*list(zip(*to_plot[fw])),
                label=fw.replace('_', ' '),
                marker=rng_cfg[fw][1],
                color=rng_cfg[fw][0],
                linestyle=rs)

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

def create_figure(entropy_plot, layers, titles):
    rows = 3
    columns = 4
    fig = plt.figure(figsize=(12,8))#, layout='tight')
    plt.rc('axes', labelsize=20)
    plt.rc('axes', titlesize=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    subfigs = fig.subfigures(nrows=rows, ncols=1)
    plt.subplots_adjust(left=0, right=1, bottom=0.1, top=1)
    fig.text(0.5, -0.05, "D", ha='center', fontsize=20)
    if entropy_plot:
        major_vert = "Shannon Entropy"
    else:
        major_vert = "Compression Ratio"
    fig.text(-0.07, 0.5, major_vert, va='center', rotation='vertical', fontsize=20)
    axes = []
    for r in range(rows):
        axes.append(subfigs[r].subplots(nrows=1, ncols=columns))
    for c in range(columns):
        for r in range(rows):
            axes[r][c].axes.set_xlim(2,8)
            axes[r][c].axes.set_xscale('log', base=2)
            if not entropy_plot:
                #axes[r][c].axes.set_aspect(0.66)
                axes[r][c].axes.set_yscale('log', base=2)
                axes[r][c].axes.set_ylim(0.9, 8)
                axes[r][c].axes.set_yticks([1, 2, 4, 8])
            else:
                axes[r][c].axes.set_ylim(0, 1)
            axes[r][c].axes.grid(visible=True)
            if c != 0:
                axes[r][c].axes.yaxis.set_ticklabels([])
            if r != rows-1:
                axes[r][c].axes.xaxis.set_ticklabels([])
            if r == 0:
                axes[r][c].set_title(titles[c])
            if c == columns-1:
                subfigs[r].supylabel(layers[r], x=1.01, fontsize=20)
    return axes, fig

def add_legend(ax):
    ax.legend(title='firmware name', ncol=4, loc='upper left', bbox_to_anchor=(0.15, -0.05), fontsize=12)
