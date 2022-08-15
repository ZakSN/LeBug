import sys
from plot_compression_results import get_ax
from plot_compression_results import parse_raw
sys.path.insert(1, '../')
from plot_results import *
import matplotlib.pyplot as plt

RESULTS_FILE="audio_compression_results.csv"
raw = read_results_file(RESULTS_FILE)
raw = parse_raw(raw)

# index order: encoder port, layer, stride, firmware
# at each index is a list of tuples of the form (D, prob_one@encoder port)
data = {}
encoder_in = 'in'
encoder_out = 'out'
for l in raw:
    if encoder_in not in data:
        data[encoder_in] = {}
    if encoder_out not in data:
        data[encoder_out] = {}
    if l[0] not in data[encoder_in]:
        data[encoder_in][l[0]] = {}
    if l[0] not in data[encoder_out]:
        data[encoder_out][l[0]] = {}
    if l[1] not in data[encoder_in][l[0]]:
        data[encoder_in][l[0]][l[1]] = {}
    if l[1] not in data[encoder_out][l[0]]:
        data[encoder_out][l[0]][l[1]] = {}
    if l[3] not in data[encoder_in][l[0]][l[1]]:
        data[encoder_in][l[0]][l[1]][l[3]] = []
    if l[3] not in data[encoder_out][l[0]][l[1]]:
        data[encoder_out][l[0]][l[1]][l[3]] = []
    data[encoder_in][l[0]][l[1]][l[3]].append((float(l[2]), entropy(float(l[5]))))
    data[encoder_out][l[0]][l[1]][l[3]].append((float(l[2]), entropy(float(l[6]))))

subfigs = create_figure(True, ['conv2d 1','activation 4','batch normalization 7'],
                              ['Stride: 0.25','Stride: 0.5','Stride: 0.75','Stride: 1.0'])
for p in data.keys():
    for l in data[p].keys():
        for s in data[p][l].keys():
            rng = data[p][l][s]
            if 'in' in p:
                style='dotted'
            else:
                style='solid'
            ax = get_ax(subfigs, l, s)
            plot_range(
                ax,
                rng,
                True,
                range_id=' ' + p,
                range_style=style)
plt.legend(title='firmware name', ncol=4, bbox_to_anchor = (1, -0.23), fontsize=12)
plt.savefig('audio_entropy_plot.png', bbox_inches='tight')
