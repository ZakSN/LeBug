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
import csv
import zlib

# class to encapsulate compression experiments
class CompressionExperiment():
    def __init__(self):
        # default emulated hardware configuration
        self.emu_cfg = {
            'N':               32,
            'M':               32,
            'IB_DEPTH':        32,
            'FUVRF_SIZE':      4,
            'VVVRF_SIZE':      8,
            'TB_SIZE':         64,
            'MAX_CHAINS':      8,
            'DATA_WIDTH':      32,
            'DELTA_SLOTS':     4,
            'DATA_TYPE':       'fixed_point',
            'BUILDING_BLOCKS': ['InputBuffer',
                                'FilterReduceUnit',
                                'VectorVectorALU',
                                'VectorScalarReduce',
                                'DataPacker',
                                'DeltaCompressor',
                                'TraceBuffer'],
        }
        self.emu_cfg['PRECISION'] = int(self.emu_cfg['DATA_WIDTH']/self.emu_cfg['DELTA_SLOTS'])
        self.emu_cfg['INV'] = twos_complement_min(self.emu_cfg['PRECISION'])

        # default end of frame bit configuration
        self.eof1_cfg = (
            False, # initial value
            True, # final value
            lambda i: i # update function
        )
        self.eof2_cfg = (
            False, # initial value
            False, # final value
            lambda i: i
        )

    #XXX need raw int and raw fxp?
    def raw(self, **kwargs):
        emu = emulatedHw(**self.emu_cfg)
        fw = firm.raw(emu.compiler)
        emu.config(fw)
        ret = {
            'emu_cfg' : self.emu_cfg,
            'emu' : emu,
            'eof1_cfg' : self.eof1_cfg,
            'eof2_cfg' : self.eof2_cfg,
        }
        return ret

    def distribution(self, **kwargs):
        emu = emulatedHw(**self.emu_cfg)
        emu.fu.vrf=list(range(self.emu_cfg['FUVRF_SIZE']*self.emu_cfg['M']))
        # can massage distribution configuration to effect compressibility
        fw = firm.distribution(emu.compiler,bins=self.emu_cfg['M'],M=self.emu_cfg['M'])
        emu.config(fw)
        ret = {
            'emu_cfg' : self.emu_cfg,
            'emu' : emu,
            'eof1_cfg' : self.eof1_cfg,
            'eof2_cfg' : self.eof2_cfg,
        }
        return ret

    # computes combined summary: 
    # sum of all elements (incompressible, unless integers)
    # and sparsity count (compressible)
    # when combined the results are not compressible
    def summary(self, **kwargs):
        emu = emulatedHw(**self.emu_cfg)
        emu.fu.vrf=list(np.concatenate((
            [0.,float('inf')],
            list(reversed(range(self.emu_cfg['FUVRF_SIZE']*self.emu_cfg['M']-2))))))
        fw = firm.summaryStats(emu.compiler)
        emu.config(fw)
        ret = {
            'emu_cfg' : self.emu_cfg,
            'emu' : emu,
            'eof1_cfg' : self.eof1_cfg,
            'eof2_cfg' : self.eof2_cfg,
        }
        return ret

    def sparsity_count(self, **kwargs):
        emu = emulatedHw(**self.emu_cfg)
        emu.fu.vrf=list(np.concatenate((
            [0.,float('inf')],
            list(reversed(range(self.emu_cfg['FUVRF_SIZE']*self.emu_cfg['M']-2))))))
        fw = firm.numSparse(emu.compiler)
        emu.config(fw)
        ret = {
            'emu_cfg' : self.emu_cfg,
            'emu' : emu,
            'eof1_cfg' : self.eof1_cfg,
            'eof2_cfg' : self.eof2_cfg,
        }
        return ret

    def spatial_sparsity(self, **kwargs):
        emu = emulatedHw(**self.emu_cfg)
        emu.fu.vrf=list(np.concatenate((
            [0.,float('inf')],
            list(reversed(range(self.emu_cfg['FUVRF_SIZE']*self.emu_cfg['M']-2))))))
        fw = firm.spatialSparsity(emu.compiler, self.emu_cfg['N'])
        emu.config(fw)
        ret = {
            'emu_cfg' : self.emu_cfg,
            'emu' : emu,
            'eof1_cfg' : self.eof1_cfg,
            'eof2_cfg' : self.eof2_cfg,
        }
        return ret

    def norm_check(self, limits=None, **kwargs):
        emu = emulatedHw(**self.emu_cfg)
        # XXX how to set fru filter values appropriately?
        emu.fu.vrf = list(np.linspace(limits[0], limits[1], 64))
        fw = firm.normCheck(emu.compiler, self.emu_cfg['M'])
        emu.config(fw)
        ret = {
            'emu_cfg' : self.emu_cfg,
            'emu' : emu,
            'eof1_cfg' : self.eof1_cfg,
            'eof2_cfg' : self.eof2_cfg,
        }
        return ret

    # I think this is a plausible control signal configuration, however in
    # practice the specific control signal manipulation may depend on how the
    # tensors are flattened
    def activation_predictiveness(self, **kwargs):
        this_emu_cfg = self.emu_cfg
        this_emu_cfg['BUILDING_BLOCKS'] = [
            'InputBuffer',
            'FilterReduceUnit',
            'VectorScalarReduce',
            'VectorVectorALU',
            'DataPacker',
            'DeltaCompressor',
            'TraceBuffer']
        emu = emulatedHw(**this_emu_cfg)
        fw = firm.activationPredictiveness(emu.compiler)
        emu.config(fw)
        this_eof1_cfg = (self.eof1_cfg[0], self.eof1_cfg[1], lambda i: ~i)
        this_eof2_cfg = (True, False, lambda i: False)
        ret = {
            'emu_cfg' : this_emu_cfg,
            'emu' : emu,
            'eof1_cfg' : this_eof1_cfg,
            'eof2_cfg' : this_eof2_cfg,
        }
        return ret

    # use zlib to compress a trace of the delta compressor's input and compute
    # the cr achieved by zlib, level sets how hard zlib tries, 0=min, 9=max
    # extremely slow!
    # step sets how frequently zlib is run, by default we only run zlib every
    # after every 100 vectors (running more frequently produces more data, but
    # is much slower)
    def calc_zlib_cr(self, dp_log, level=9, logfile=None, step=1):
        def cr(s):
            s = s.tobytes()
            ucl = len(s)
            s = zlib.compress(s, level=9)
            cl = len(s)
            return ucl/cl
        stream = None
        num_vec = 0
        for i in range(len(dp_log)):
            if dp_log[i][1] == 1:
                num_vec += 1
                if stream is None:
                    stream = dp_log[i][0]
                else:
                    stream = np.vstack((stream, dp_log[i][0]))
                if (logfile is not None) and (num_vec % step == 0):
                    partial_cr = cr(stream)
                    with open(logfile, 'a') as file:
                        file.write(str(num_vec)+','+str(partial_cr)+'\n')

        return cr(stream)

    # missing total invalidity
    def run_experiment(self, istream, D, firmware, experimental_cfg, q, zlib=[None, None], early_stop=None):
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
                if early_stop is not None:
                    if emu.dc.num_vec >= early_stop:
                        break
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
        bpv = self.emu_cfg['N']*self.emu_cfg['DATA_WIDTH']
        cr = tuobj.compression_ratio(v_nodata, log['tb'][-1][0], decomp_tb, bpv)

        # compute input/output bit 1 probabability for entropy measurement
        # for the output we also need to consider any bits left in the last_reg
        pi = emu.dc.bi/emu.dc.ti
        po = (emu.dc.bo + emu.dc.pop_count(emu.dc.last_reg))/(emu.dc.to + (emu.N*emu.DATA_WIDTH))

        if zlib is not None:
            zcr = self.calc_zlib_cr(log['dp'][:], level=zlib[0], logfile=zlib[1])
        else:
            zcr = -1

        q.put((*experimental_cfg, cr, pi, po, zcr))

# reshape all frames in an input tensor to a matrix that is N elements wide,
# padding any excess elements with 0
def prepare_data_frames(layer_data, N, asint=False):
    indata = []
    for frame_idx in range(layer_data.shape[0]):
        frame = layer_data[frame_idx][0]
        frame = frame.flatten()
        frame.resize((math.ceil(frame.shape[0]/N), N))
        if asint:
            frame = frame.astype(int)
        indata.append(frame)
    return np.array(indata)

# get all of the all ready completed configurations from the results file
def get_all_ready_done(RESULTS_FILE, num_results=4):
    all_ready_done = []
    try:
        with open(RESULTS_FILE, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                all_ready_done.append(tuple(row[0:-1*num_results]))
    except FileNotFoundError:
        pass
    return all_ready_done

# run all of the experiements
# we run cpu_count() - 1 threads in parallel, since there's a lot of experiements
def multi_run(proc, RESULTS_FILE, q):
    proc = [multiprocessing.Process(target=p[0], args=p[1]) for p in proc]
    total = len(proc)
    print("Running "+str(total)+" Experiments")

    finished = 0
    running = []
    while (len(proc) > 0) or (finished < total):
        # determine the maximum number of CPUs (leaving one idle on multicore machines):
        num_cpu = max(1, multiprocessing.cpu_count() - 1)
        # if less than the maximum number of threads are running, start some more
        while (len(running) < num_cpu) and (len(proc) > 0):
            running.append(proc.pop())
            running[-1].start()

        # try to join any threads that are finished
        for p in running:
            p.join(timeout=1)
            if not p.is_alive():
                running.remove(p)
                finished = finished + 1
                print("Finished "+str(finished)+" of "+str(total)+" experiments")

        # periodically empty some results from the queue incase we crash mid run
        if q.qsize() >= int(total/10):
            with open(RESULTS_FILE, 'a') as file:
                for _ in range(int(total/10)):
                    if not q.empty():
                        file.write(','.join(map(str, q.get())) + '\n')

    # dump anything that was left in the queue
    with open(RESULTS_FILE, 'a') as file:
        while not q.empty():
            file.write(','.join(map(str, q.get())) + '\n')
