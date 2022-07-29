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
import pickle

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

    # missing total invalidity

    def run_experiment(emu_cfg, emu, indata, num_frames, eof1_cfg, eof2_cfg, sampling_frequency, verbose=None):
        # verbose print
        def v_print(*args):
            if verbose is not None:
                print(*args)

        # step through the frames in indata
        # each frame is assumed to be an arbitrary number of vectors long, with
        # each vector being N elemtents wide
        for frame_idx in range(0, num_frames, sampling_frequency):
            v_print("Processing frame "+str(frame_idx)+" of "+str(num_frames))
            inframe = indata[frame_idx]

            # feed the frame into the input buffer
            vidx = 0
            eof1 = eof1_cfg[0]
            eof2 = eof2_cfg[0]
            frame_stop = inframe.shape[0] - 1
            while vidx < frame_stop:
                # fill the input buffer as long as there are still vectors to feed in
                for ibidx in range(emu_cfg['IB_DEPTH']):
                    emu.push([inframe[vidx][:], eof1, eof2])
                    eof1 = eof1_cfg[2](eof1)
                    eof2 = eof2_cfg[2](eof2)
                    vidx = vidx + 1
                    if vidx >= frame_stop:
                        break
                # drain the input buffer
                while len(emu.ib.buffer) > 0:
                    s = len(emu.ib.buffer)
                    log = emu.run(steps = s)
            # the last vector in a frame is a special case since the end-of-frame
            # bits may need to be set
            emu.push([inframe[-1][:], eof1_cfg[1], eof2_cfg[1]])
            log = emu.run(steps=10)

        # configure the decompression algorithm
        dd = DeltaDecompressor(
            emu_cfg['N'],
            emu_cfg['DATA_WIDTH'],
            emu_cfg['DELTA_SLOTS'],
            emu_cfg['TB_SIZE'])

        # decompress (and decode in fxp) the trace buffer
        v_print(log['tb'][-1][0][:][:])
        decomp_tb = dd.decompress(log['dc'][-1][1], log['tb'][-1][0], log['tb'][-1][1], log['tb'][-1][2])
        # decoding results isn't required for calculating compression ratios, and
        # is firmware dependant (e.g. fxp data may be used to calculate int results)
        # thus the below is only useful for debugging
        '''
        if emu_cfg['DATA_TYPE'] == 'fixed_point':
            decomp_tb = np.array(encodedIntTofloat(decomp_tb,emu_cfg['DATA_WIDTH']))
        '''
        v_print(decomp_tb)
        v_print(decomp_tb.shape)

        # compute the compression ratio
        tuobj = TestUtils()
        nodata = n_bit_nodata(emu_cfg['DELTA_SLOTS'], emu_cfg['PRECISION'], emu_cfg['INV'])
        v_nodata = np.full((1, emu_cfg['N']), nodata)
        cr = tuobj.compression_ratio(v_nodata, log['tb'][-1][0], decomp_tb)
        v_print(cr)
        return cr

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

# create a function to encapsulate the inner experiment loops
# this sweeps firmware and number of delta slots, and produces a pickled dictionary
def experiment_process(sampling_frequency, layer, ce, results_directory):
    results_file = str(sampling_frequency) + "_" +  layer[0] + ".pickle"
    if os.path.exists(os.path.join(results_directory, results_file)):
        print("Results for: " + results_file + " all ready exist. Skipping.")
        return
    indata = layer[1]
    results = {}
    args = {'limits':(np.min(indata), np.max(indata))}
    for firmware in [ce.raw, ce.distribution, ce.sparsity_count, ce.spatial_sparsity, ce.norm_check, ce.activation_predictiveness]:
        for delta_slots in [2, 4, 8, 16]:
            # configure the emulator
            ce.emu_cfg['DELTA_SLOTS'] = delta_slots
            ce.emu_cfg['PRECISION'] = int(ce.emu_cfg['DATA_WIDTH']/ce.emu_cfg['DELTA_SLOTS'])
            ce.emu_cfg['INV'] = twos_complement_min(ce.emu_cfg['PRECISION'])

            # setup the experiment
            experiment = firmware(**args)
            experiment['indata'] = indata
            experiment['num_frames'] = indata.shape[0]
            experiment['sampling_frequency'] = sampling_frequency
            #experiment['verbose'] = True

            # get the compression ratio
            cr = CompressionExperiment.run_experiment(**experiment)

            # add the compression ratio to the appropriate range
            if firmware.__name__ not in results:
                results[firmware.__name__] = []
            results[firmware.__name__].append((delta_slots, cr))

    # pickle the results for later consumption by the plotting script
    with open(os.path.join(results_directory, results_file), 'wb') as file:
        pickle.dump(results, file)

for input_video in ["face", "noface"]:
    # create a list of tensors to use as experimental input
    INPUT_TENSOR_DIR = os.path.join("input_tensors", input_video)
    input_tensors = []
    for tensor in [t for t in os.listdir(INPUT_TENSOR_DIR) if t.split('.')[1] == 'npy']:
        input_tensors.append((
            tensor.split('.')[0],
            prepare_data_frames(np.load(os.path.join(INPUT_TENSOR_DIR, tensor)), 32)))

    # we don't report the results for some layers we sampled, this line filters
    # out those inputs to speed up experimental runtime. comment it out to get
    # results on all layers.
    input_tensors = [tensor for tensor in input_tensors if ('input' not in tensor[0]) and ('dense' not in tensor[0])]

    # create a new experiment
    ce = CompressionExperiment()

    # create a directory to store experimental results
    RESULTS_DIRECTORY = input_video + "_" + "results"
    try:
        if not os.path.exists(RESULTS_DIRECTORY):
            os.makedirs(RESULTS_DIRECTORY)
    except OSError:
        print('ERROR: could not make results directory')
        exit()

    # run the experiment (in parallel so it hopefully goes a little faster)
    proc = []
    for sampling_frequency in [1, 2, 4, 8]:
        for layer in input_tensors:
            p = multiprocessing.Process(
                target=experiment_process,
                args=(sampling_frequency, layer, ce, RESULTS_DIRECTORY))
            proc.append(p)

    print("Running " + str(len(proc)) + " experiments")

    # set off all of the experiments
    for p in proc:
        p.start()

    still_running = True
    while still_running:
        still_running = False
        for p in proc:
            p.join(timeout=1)
            if p.is_alive():
                still_running = True
