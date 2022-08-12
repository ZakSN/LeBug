import sys
sys.path.insert(1, '../')
from plot_results import *
import matplotlib.pyplot as plt

RESULTS_FILE="video_compression_results.csv"
raw = read_results_file(RESULTS_FILE)
subfigs = create_figure(False)

# index order: video input, layer, sampling period, firmware
# at each index is a list of tuples of the form (D, compression_ratio)
data = {}
for l in raw:
    if l[0] not in data: # input
        data[l[0]] = {}
    if l[1] not in data[l[0]]: # layer
        data[l[0]][l[1]] = {}
    if l[2] not in data[l[0]][l[1]]:
        data[l[0]][l[1]][l[2]] = {}
    if l[4] not in data[l[0]][l[1]][l[2]]:
        data[l[0]][l[1]][l[2]][l[4]] = []
    data[l[0]][l[1]][l[2]][l[4]].append((float(l[3]), float(l[5])))

def get_ax(sf, l, s):
    def row_lut(k):
        if '6' in k:
            return 0
        if '13' in k:
            return 1
        if '20' in k:
            return 2
    def column_lut(k):
        if '1' in s:
            return 0
        if '2' in s:
            return 1
        if '4' in s:
            return 2
        if '8' in s:
            return 3
    return sf[row_lut(l)][column_lut(s)]

for v in data.keys():
    for l in data[v].keys():
        for s in data[v][l].keys():
            rng = data[v][l][s]
            if 'no' in v:
                style='dotted'
            else:
                style='solid'
            ax = get_ax(subfigs, l, s)
            plot_range(
                ax,
                rng,
                False,
                range_id=' ' + v,
                range_style=style)

plt.legend(title='firmware name', ncol=4, bbox_to_anchor = (1, -0.23), fontsize=12)
plt.savefig('video_compression_plot.png', bbox_inches='tight')

