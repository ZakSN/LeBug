import numpy as np
import sys
sys.path.insert(1, '../../../src')
from emulator.emulator import emulatedHw
from software.delta_decompressor import DeltaDecompressor
import firmware.firmware as firm
sys.path.insert(1, '../../../examples')
from test_utils import TestUtils
from misc.misc import *
import matplotlib.pyplot as plt

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

    def activation_predictiveness(self, **kwargs):
        # should be run on activation data...
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
        this_eof2_cfg = (self.eof2_cfg[0], True, self.eof2_cfg[2])
        ret = {
            'emu_cfg' : this_emu_cfg,
            'emu' : emu,
            'eof1_cfg' : this_eof1_cfg,
            'eof2_cfg' : this_eof2_cfg,
        }
        return ret

    # missing total invalidity

    def run_experiment(emu_cfg, emu, indata, num_frames, eof1_cfg, eof2_cfg, verbose=None):
        # verbose print
        def v_print(*args):
            if verbose is not None:
                print(*args)

        # step through the frames in indata
        # each frame is assumed to be an arbitrary number of vectors long, with
        # each vector being N elemtents wide
        for frame_idx in range(num_frames):
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
        if emu_cfg['DATA_TYPE'] == 'fixed_point':
            decomp_tb = np.array(encodedIntTofloat(decomp_tb,emu_cfg['DATA_WIDTH']))

        v_print(decomp_tb)
        v_print(decomp_tb.shape)

        # compute the compression ratio
        tuobj = TestUtils()
        nodata = n_bit_nodata(emu_cfg['DELTA_SLOTS'], emu_cfg['PRECISION'], emu_cfg['INV'])
        v_nodata = np.full((1, emu_cfg['N']), nodata)
        cr = tuobj.compression_ratio(v_nodata, log['tb'][-1][0], decomp_tb)
        v_print(cr)
        return cr

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

# load the layer data
raw = np.load("output_file.npy")

# slice and concatenate the raw layer data into regular frames
indata = []
for frame_idx in range(raw.shape[0]):
    frame = raw[frame_idx][0][0][:][:]
    for i in range(1, raw.shape[2]):
        frame = np.vstack((raw[frame_idx][0][i][:][:], frame))
    indata.append(frame)
indata = np.array(indata)

# create a new experiment
ce = CompressionExperiment()

# XXX setting data type to int
ce.emu_cfg['DATA_TYPE'] = 'int'
indata = indata.astype(int)

results = {}

# sweep number of delta slots for each experiment
args = {'limits':(np.min(indata), np.max(indata))}
with open("compression_experiment.csv", 'w') as file:
    file.write('firmware, delta slots, compression ratio\n')
    for apparatus in [ce.distribution, ce.summary, ce.spatial_sparsity, ce.norm_check, ce.activation_predictiveness]:
        for delta_slots in [2, 4, 8, 16]:
            ce.emu_cfg['DELTA_SLOTS'] = delta_slots
            ce.emu_cfg['PRECISION'] = int(ce.emu_cfg['DATA_WIDTH']/ce.emu_cfg['DELTA_SLOTS'])
            ce.emu_cfg['INV'] = twos_complement_min(ce.emu_cfg['PRECISION'])

            experiment = apparatus(**args)
            experiment['indata'] = indata
            experiment['num_frames'] = indata.shape[0]
            #experiment['verbose'] = True
            cr = CompressionExperiment.run_experiment(**experiment)
            result = apparatus.__name__ + ", "+ str(delta_slots) + ", " + str(cr)
            file.write(result+'\n')
            print(result)
            if apparatus.__name__ not in results:
                results[apparatus.__name__] = []
            results[apparatus.__name__].append((delta_slots, cr))

plot_results(results)
