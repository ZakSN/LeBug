import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import os
import sys
sys.path.insert(1, '../')
from plot_results import read_results_file

RESULTS_FILE = "zlib_video_compression_results.csv"

raw = read_results_file(RESULTS_FILE)

results = {}
for x in raw:
    if x[4] not in results:
        results[x[4]] = []
    results[x[4]].append(float(x[-1]))

for fw in results:
    results[fw] = sum(results[fw])/len(results[fw])

ranges = []
for fw in results:
    ranges.append((fw, results[fw]))

plt.bar(*list(zip(*ranges)))
plt.xticks(rotation = 90)
plt.grid(visible=True, which='both', axis='y')
ax = plt.gca()
ax.set_yscale('log')
ax.yaxis.set_minor_formatter(FormatStrFormatter('%d'))
plt.show()

print(results)
