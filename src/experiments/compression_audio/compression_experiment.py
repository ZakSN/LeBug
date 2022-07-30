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
import multiprocessing
from multiprocessing import Queue
import pickle
from experiments.abstract_compression_experiment import AbstractCompressionExperiment
from experiments.abstract_compression_experiment import prepare_data_frames

class CompressionExperiment(AbstractCompressionExperiment):
    def run_experiment(self, data_frames, D, firmware, cfg, q):
        # configure the emulator
        self.emu_cfg['DELTA_SLOTS'] = d
        self.emu_cfg['PRECISION'] = int(self.emu_cfg['DATA_WIDTH']/self.emu_cfg['DELTA_SLOTS'])
        self.emu_cfg['INV'] = twos_complement_min(self.emu_cfg['PRECISION'])

        exp = firmware(limits=(np.min(data_frames), np.max(data_frames)))

        for f in data_frames: # for each data frame in the flattened tensor stream
            # feed the vectors into the emulator
            frame_stop = f.shape[0]-1
            vidx = 0
            emu = exp['emu']
            eof1 = exp['eof1_cfg'][0]
            eof2 = exp['eof2_cfg'][0]
            while vidx < frame_stop:
                # fill the input buffer
                for ibidx in range(self.emu_cfg['IB_DEPTH']):
                    emu.push([f[vidx][:], eof1, eof2])
                    eof1 = exp['eof1_cfg'][2](eof1)
                    eof2 = exp['eof2_cfg'][2](eof2)
                    vidx = vidx + 1
                    if vidx >= frame_stop:
                        break
                # drain the input buffer
                while len(emu.ib.buffer) > 0:
                    s = len(emu.ib.buffer)
                    emu.run(steps = s)
            # last vector is special
            emu.push([f[-1][:], exp['eof1_cfg'][1], exp['eof2_cfg'][1]])
            log = emu.run(steps = 10)

        # configure the decompression algorithm
        dd = DeltaDecompressor(
            self.emu_cfg['N'],
            self.emu_cfg['DATA_WIDTH'],
            self.emu_cfg['DELTA_SLOTS'],
            self.emu_cfg['TB_SIZE'])

        # decompress (and decode in fxp) the trace buffer
        decomp_tb = dd.decompress(log['dc'][-1][1], log['tb'][-1][0], log['tb'][-1][1], log['tb'][-1][2])

        # compute the compression ratio
        tuobj = TestUtils()
        nodata = n_bit_nodata(self.emu_cfg['DELTA_SLOTS'], self.emu_cfg['PRECISION'], self.emu_cfg['INV'])
        v_nodata = np.full((1, self.emu_cfg['N']), nodata)
        cr = tuobj.compression_ratio(v_nodata, log['tb'][-1][0], decomp_tb)
        q.put((*cfg, cr))
        #return (*cfg, cr)

INPUT_TENSOR_DIR = os.path.join('input_tensors', 'tfds_speech_commands_10clip_stream')
RESULTS_FILE = 'compression_audio_results.txt'
# use sets so that we don't get multiple instances of each value
layers = set()
strides = set()
clip_lengths = set()
delta_slots = [2, 4, 8, 16]
ce = CompressionExperiment()
firmwares = [ce.raw, ce.distribution, ce.sparsity_count, ce.spatial_sparsity, ce.norm_check, ce.activation_predictiveness]
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

proc = []
q = Queue()

# setup all of our experiments
for l in layers:
    for s in strides:
        for c in clip_lengths:
            for d in delta_slots:
                for f in firmwares:
                    path = l+"_"+s+"_"+c+".npy"
                    df = prepare_data_frames(np.load(os.path.join(INPUT_TENSOR_DIR, path)), N)
                    p = multiprocessing.Process(
                        target=ce.run_experiment,
                        args=(df, d, f, (l, s, c, d, f.__name__), q))
                    proc.append(p)

# run the experiments
print("Running "+str(len(proc))+" experiments")

for p in proc:
    p.start()

still_running = True
while still_running:
    still_running = False
    for p in proc:
        p.join(timeout=1)
        if p.is_alive():
            still_running = True

# when the experiments are finished dump the results queue to a file
with open(RESULTS_FILE, 'w') as file:
    while not q.empty():
        file.write(', '.join(map(str, q.get())) + '\n')
