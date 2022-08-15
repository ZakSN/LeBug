import sys
sys.path.insert(1, '../')
from plot_results import *
import matplotlib.pyplot as plt

RESULTS_FILE="audio_compression_results.csv"
raw = read_results_file(RESULTS_FILE)

def parse_raw(raw):
    parsed = []
    for l in raw:
        parsed.append((l[0], float(l[1])/float(l[2]), l[3], l[4], l[5], l[6], l[7]))
    return parsed

parsed = parse_raw(raw)

# index order: layer, stride, firmware
# at each index is a list of tuples of the form (D, compression_ratio)
data = {}
for l in parsed:
    if l[0] not in data:
        data[l[0]] = {}
    if l[1] not in data[l[0]]:
        data[l[0]][l[1]] = {}
    if l[3] not in data[l[0]][l[1]]:
        data[l[0]][l[1]][l[3]] = []
    data[l[0]][l[1]][l[3]].append((float(l[2]), float(l[4])))

subfigs = create_figure(False, ['conv2d 1','activation 4','batch normalization 7'],
                              ['Stride: 0.25','Stride: 0.5','Stride: 0.75','Stride: 1.0'])

def get_ax(sf, l, s):
    def row_lut(k):
        if 'conv' in k:
            return 0
        if 'activation' in k:
            return 1
        if 'batch' in k:
            return 2
    def column_lut(k):
        if 0.25 == s:
            return 0
        if 0.50 == s:
            return 1
        if 0.75 == s:
            return 2
        if 1.00 == s:
            return 3
    return sf[row_lut(l)][column_lut(s)]

for l in data.keys():
    for s in data[l].keys():
        rng = data[l][s]
        rng['ideal'] = [(2,2*(64/65)), (4,4*(64/65)), (8,8*(64/65)), (16,16*(64/65))]
        ax = get_ax(subfigs, l, s)
        plot_range(
            ax,
            rng,
            False)

plt.legend(title='firmware name', ncol=4, bbox_to_anchor = (0.5, -0.23), fontsize=12)
plt.savefig('audio_compression_plot.png', bbox_inches='tight')
