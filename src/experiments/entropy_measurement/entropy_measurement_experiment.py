import numpy as np
import os
import sys
sys.path.insert(1, '../../../src')
from emulator.emulator import emulatedHw
import firmware.firmware as firm
sys.path.insert(1, '../../../examples')
from misc.misc import *
import math
import multiprocessing
from multiprocessing import Queue
import csv
from experiments.compression_experiment import CompressionExperiment
from experiments.compression_experiment import prepare_data_frames
from experiments.compression_experiment import multi_run
from experiments.compression_experiment import get_all_ready_done


class EntropyExperiment(CompressionExperiment):
    def build_symbol_stream(self, log):
        stream = None
        for i in range(len(log['dp'])):
            if log['dp'][i][1]:
                if stream is None:
                    stream = log['dp'][i][0]
                else:
                    stream = np.vstack((stream, log['dp'][i][0]))
        return stream.tobytes()

    def count_symbol_blocks(self, stream, n):
        counts = {}
        for x in range(len(stream)-n):
            idx = stream[x:x+n]
            if idx not in counts:
                counts[idx]=1
            else:
                counts[idx]+=1
        return counts

    def compute_Gn(self, counts):
        total = sum(counts.values())
        Gn = 0
        for i in counts.keys():
            p_i = counts[i]/total
            if p_i == 0:
                Gn+=0
            else:
                Gn+=(-p_i)*math.log(p_i,2)
        return Gn

    def run_experiment(self, istream, D, firmware, experimental_cfg, q, max_block_size):
        # configure the emulator
        self.emu_cfg['DELTA_SLOTS'] = D
        self.emu_cfg['PRECISION'] = int(self.emu_cfg['DATA_WIDTH']/self.emu_cfg['DELTA_SLOTS'])
        self.emu_cfg['INV'] = twos_complement_min(self.emu_cfg['PRECISION'])

        exp = firmware(self, limits=(np.min(istream), np.max(istream)))
        emu = exp['emu']
        emu.dc.measure_entropy = True

        for f in istream: # for each frame in the input tensor stream
            # feed the vectors into the emulator
            frame_stop = f.shape[0]-1
            vidx = 0
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

        # compute partial entropies of the current log
        # i.e. theorem 5 from A mathematical theory of communication
        stream = self.build_symbol_stream(log)
        Hdbg = []
        for n in range(1, max_block_size):
            counts = self.count_symbol_blocks(stream, n)
            Hn = self.compute_Gn(counts)/n
            Hdbg.append(Hn)

        q.put((*experimental_cfg, Hdbg))


INPUT_TENSOR_DIR = os.path.join('..', 'compression_video', 'input_tensors')
RESULTS_FILE = 'entropy_measurement_results.csv'

firmwares = [CompressionExperiment.raw,
             CompressionExperiment.distribution,
             CompressionExperiment.sparsity_count,
             CompressionExperiment.spatial_sparsity,
             CompressionExperiment.norm_check,
             CompressionExperiment.activation_predictiveness]
N = 32

def build_proc_list(INPUT_TENSOR_DIR, RESULTS_FILE, firmwares, N):
    # set fixed parameters
    v = 'face' # name of input video
    l = 'activation_6.npy' # layer name
    s = 8 # sampling period
    d = 2 # number of delta slots
    max_block_size = 500

    all_ready_done = get_all_ready_done(RESULTS_FILE, num_results=max_block_size-1)

    # load and flatten tensor stream
    tensor_stream = np.load(os.path.join(INPUT_TENSOR_DIR, v, l))
    tensor_stream = prepare_data_frames(tensor_stream, N)
    tensor_stream = tensor_stream[0::s]

    proc = []
    q = Queue()

    for f in firmwares:
        experimental_cfg = (v,l.split('.')[0],s,d,f.__name__)
        if tuple(map(str, experimental_cfg)) not in all_ready_done:
            ee = EntropyExperiment()
            proc.append((ee.run_experiment, (tensor_stream, d, f, experimental_cfg, q, max_block_size)))

    return proc, q

if __name__ == "__main__":
    proc, q = build_proc_list(INPUT_TENSOR_DIR,
                              RESULTS_FILE,
                              firmwares,
                              N)
    multi_run(proc, RESULTS_FILE, q)
