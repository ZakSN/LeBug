import numpy as np
import os
import sys
sys.path.insert(1, '../../../src')
from emulator.emulator import emulatedHw
from software.delta_decompressor import DeltaDecompressor
import firmware.firmware as firm
sys.path.insert(1, '../../../examples')
from test_utils import TestUtils
from misc.misc import *
import math
from multiprocessing import Queue
from experiments.compression_experiment import CompressionExperiment
from experiments.compression_experiment import prepare_data_frames
from experiments.compression_experiment import multi_run
from experiments.compression_experiment import get_all_ready_done

INPUT_TENSOR_DIR = os.path.join('input_tensors', 'tfds_speech_commands_10clip_stream')
RESULTS_FILE = 'compression_audio_results.csv'

# use sets so that we don't get multiple instances of each value
layers = set()
strides = set()
clip_lengths = set()
delta_slots = [2, 4, 8]
firmwares = [CompressionExperiment.raw,
             CompressionExperiment.distribution,
             CompressionExperiment.sparsity_count,
             CompressionExperiment.spatial_sparsity,
             CompressionExperiment.norm_check,
             CompressionExperiment.activation_predictiveness]
N = 32

for datafile in [d for d in os.listdir(INPUT_TENSOR_DIR) if d.split('.')[-1] == 'npy']:
    datafile = datafile.split('.')[0]
    datafile = datafile.split('_')

    # populate our experimental sweeps based on what we find in the input
    # tensor directory
    clip_lengths.add(datafile[-1])
    strides.add(datafile[-2])
    layers.add("_".join(datafile[0:-2]))

# convert sets to lists
layers = list(layers)
strides = list(strides)
clip_lengths = list(clip_lengths)

# filter data we don't want to process
# XXX remove these lines to get more output (takes longer to run)
layers = [l for l in layers if ('dense' not in l) and ('input' not in l)]

all_ready_done = get_all_ready_done(RESULTS_FILE)

proc = []
q = Queue()

# setup all of our experiments
for l in layers:
    for s in strides:
        for c in clip_lengths:
            path = l+"_"+s+"_"+c+".npy"
            stream = prepare_data_frames(np.load(os.path.join(INPUT_TENSOR_DIR, path)), N)
            for d in delta_slots:
                for f in firmwares:
                    experimental_cfg = (l, s, c, d, f.__name__)
                    if tuple(map(str, experimental_cfg)) not in all_ready_done:
                        ce = CompressionExperiment()
                        proc.append((ce.run_experiment, (stream, d, f, experimental_cfg, q)))

multi_run(proc, RESULTS_FILE, q)
