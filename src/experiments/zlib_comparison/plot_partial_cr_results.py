import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(1, '../')
from plot_results import read_results_file

RESULTS_DIR = "partial_cr_results"

results = {}
for rf in os.listdir(RESULTS_DIR):
    key = rf.split('.')[0].replace('_', ' ')
    results[key] = read_results_file(os.path.join(RESULTS_DIR, rf))
    results[key] = list(map(lambda x: (float(x[0]), float(x[1])), results[key]))
    results[key] = np.array(results[key])

'''
for k in results:
    results[k] = results[k][0::100,:]
'''

markers = ['>', '+', '.', ',', 'o', 'v', 'x', 'X', 'D', '|']
i = 0

for k in results:
    #if ('spatial' not in k) and ('raw' not in k):
    plt.semilogx(*list(zip(*results[k])), label=k, marker=markers[i%len(markers)])
    i += 1

plt.grid(visible=True, which='both')
plt.xlabel("Number of Vectors")
plt.ylabel("Zlib Compression Ratio")
plt.legend(bbox_to_anchor=(1,1), loc="upper left")
plt.savefig('zlib_iterative_compression.png', bbox_inches='tight')
