import matplotlib.pyplot as plt

# draw a plot
def plot_results(results):
    markers = ['>', '+', 'o', 'v', 'x', 'X', 'D', '|']
    for idx, key in enumerate(results):
        plt.plot(*list(zip(*results[key])), label=key.replace('_', ' '), marker=markers[idx%8])
    plt.plot((2, 4, 8, 16), (2, 4, 8, 16), linestyle='dotted', color='black', label="Ideal")
    plt.title("Compression Ratio vs. DELTA_SLOTS")
    plt.xlabel("DELTA_SLOTS [# $\delta$s to compress]")
    plt.ylabel("Compression Ratio [(# vectors)/(# TB addrs)]")
    plt.xticks(np.logspace(1, 4, num=4, base=2))
    plt.yticks(np.arange(0, 17, 1))
    plt.legend(title='Firmware name')
    plt.grid(visible=True)
    plt.savefig("compression_experiment.png")
    plt.show()
