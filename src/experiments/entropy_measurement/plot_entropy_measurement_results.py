import sys
import numpy as np
sys.path.insert(1, '../')
from plot_results import *
import matplotlib.pyplot as plt

RESULTS_FILE="entropy_measurement_results.csv"
raw = read_results_file(RESULTS_FILE)

markers = ['>', '+', 'o', 'v', 'X', 'x', '.']

data = []
for x in raw:
    data.append([' '.join(x[0:5]), ''.join(x[5:-1])])
    data[-1][1] = data[-1][1].replace('[', '')
    data[-1][1] = data[-1][1].replace(']', '')
    data[-1][1] = list(map(float, data[-1][1].split()))

for idx, d in enumerate(data):
    plt.plot(np.arange(1, len(d[1])+1), d[1], label = d[0])#, marker=markers[idx%len(markers)])
plt.legend()
plt.xlabel("Block size [# bytes]")
plt.ylabel("Approximate Entropy [bits/byte]")
plt.title("Approximate Entropy vs. Block size")
plt.ylim(0, 3)
plt.xlim(0,len(data[0][1]))
plt.minorticks_on()
plt.grid(visible=True, which='minor', axis='y')
plt.grid(visible=True, which='major', axis='y', color='dimgrey')
plt.grid(visible=True, which='major', axis='x', color='dimgrey')
plt.savefig("entropy_measurement_plot.png", bbox_inches='tight')
