import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import os
import sys
sys.path.insert(1, '../')
from plot_results import read_results_file

def get_avg_cr(RESULTS_FILE, cr_col):
    raw = read_results_file(RESULTS_FILE)

    results = {}
    for x in raw:
        fw = x[4].replace('_', '\n')
        if fw not in results:
            results[fw] = []
        results[fw].append(float(x[cr_col]))

    for fw in results:
        results[fw] = sum(results[fw])/len(results[fw])

    ranges = []
    for fw in results:
        ranges.append((fw, results[fw]))
    return list(zip(*ranges))

zlib_ranges = get_avg_cr('zlib_video_compression_results.csv', -1)
video_ranges = get_avg_cr(os.path.join('..','compression_video','video_compression_results.csv'), 5)

width = 0.4
plt.bar(np.arange(len(zlib_ranges[0])), zlib_ranges[1], width=width, label="Zlib")
plt.bar(np.arange(len(video_ranges[0])) + width, video_ranges[1], width=width, label="Delta Compressor")
plt.xticks(np.arange(len(video_ranges[0])), video_ranges[0], rotation = 45)
plt.grid(visible=True, which='both', axis='y')
plt.legend()
plt.xlabel("Firmware Name")
plt.ylabel("Compression Ratio (log scale)")
plt.title("Average Compression Ratio vs. Firmware")
ax = plt.gca()
ax.set_yscale('log')
ax.yaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
plt.gcf().set_size_inches(8, 8)
plt.savefig("cr_vs_fw.png", bbox_inches='tight')
