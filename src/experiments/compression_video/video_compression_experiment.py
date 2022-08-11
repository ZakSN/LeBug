import numpy as np
import os
import sys
sys.path.insert(1, '../../../src')
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

INPUT_TENSOR_DIR = 'input_tensors'
RESULTS_FILE = 'video_compression_results.csv'

input_videos = ['face', 'noface']
layers = []
sampling_period = [1, 2, 4, 8]
delta_slots = [2, 4, 8]
firmwares = [CompressionExperiment.raw,
             CompressionExperiment.distribution,
             CompressionExperiment.sparsity_count,
             CompressionExperiment.spatial_sparsity,
             CompressionExperiment.norm_check,
             CompressionExperiment.activation_predictiveness]
N = 32

all_ready_done = get_all_ready_done(RESULTS_FILE)

proc = []
q = Queue()

for v in input_videos:
    # get the layers from the names of the input tensor files
    layers = [i for i in os.listdir(os.path.join(INPUT_TENSOR_DIR, v)) if '.npy' in i]
    # filter out layer data we don't want to plot
    layers = [i for i in layers if ('input' not in i) and ('dense' not in i)]
    for l in layers:
        # load the required tensor stream
        tensor_stream = np.load(os.path.join(INPUT_TENSOR_DIR, v, l))
        # flatten the tensors in the stream
        tensor_stream = prepare_data_frames(tensor_stream, N)
        for s in sampling_period:
            # subsample the tensor stream
            resampled_stream = tensor_stream[0::s]
            for d in delta_slots:
                for f in firmwares:
                    experimental_cfg = (v,l.split('.')[0],s,d,f.__name__)
                    if tuple(map(str, experimental_cfg)) not in all_ready_done:
                        ce = CompressionExperiment()
                        proc.append((ce.run_experiment, (resampled_stream, d, f, experimental_cfg, q)))

multi_run(proc, RESULTS_FILE, q)
