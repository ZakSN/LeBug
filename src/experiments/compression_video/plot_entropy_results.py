import sys
from plot_compression_results import get_ax
sys.path.insert(1, '../')
from plot_results import *
import matplotlib.pyplot as plt

RESULTS_FILE="video_compression_results.csv"
raw = read_results_file(RESULTS_FILE)

# index order: video input, encoder port, layer, sampling period, firmware
# at each index is a list of tuples of the form (D, prob_one@encoder port)
data = {}
encoder_in = 'in'
encoder_out = 'out'
for l in raw:
    if l[0] not in data:
        data[l[0]] = {}
    if encoder_in not in data[l[0]]:
        data[l[0]][encoder_in] = {}
    if encoder_out not in data[l[0]]:
        data[l[0]][encoder_out] = {}
    if l[1] not in data[l[0]][encoder_in]:
        data[l[0]][encoder_in][l[1]] = {}
    if l[1] not in data[l[0]][encoder_out]:
        data[l[0]][encoder_out][l[1]] = {}
    if l[2] not in data[l[0]][encoder_in][l[1]]:
        data[l[0]][encoder_in][l[1]][l[2]] = {}
    if l[2] not in data[l[0]][encoder_out][l[1]]:
        data[l[0]][encoder_out][l[1]][l[2]] = {}
    if l[4] not in data[l[0]][encoder_in][l[1]][l[2]]:
        data[l[0]][encoder_in][l[1]][l[2]][l[4]] = []
    if l[4] not in data[l[0]][encoder_out][l[1]][l[2]]:
        data[l[0]][encoder_out][l[1]][l[2]][l[4]] = []
    data[l[0]][encoder_in][l[1]][l[2]][l[4]].append((float(l[3]), entropy(float(l[6]))))
    data[l[0]][encoder_out][l[1]][l[2]][l[4]].append((float(l[3]), entropy(float(l[7]))))

for v in data.keys():
    subfigs = create_figure(True,
                            ['activation 6', 'activation 13', 'activation 20'],
                            ['Sampling Period: 1', 'Sampling Period 2', 'Sampling Period: 4', 'Sampling Period: 8'])
    for p in data[v].keys():
        for l in data[v][p].keys():
            for s in data[v][p][l].keys():
                rng = data[v][p][l][s]
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
    plt.savefig('video_entropy_plot_'+str(v)+'.png', bbox_inches='tight')
