import numpy as np
import sys
sys.path.insert(1, '../../../src')
from emulator.emulator import emulatedHw
from software.delta_decompressor import DeltaDecompressor
import firmware.firmware as firm
sys.path.insert(1, '../../../examples')
from test_utils import TestUtils
from misc.misc import *

def distribution(emu_cfg, **kwargs):
    proc = emulatedHw(**emu_cfg)
    proc.fu.vrf=list(range(emu_cfg['FUVRF_SIZE']*emu_cfg['M']))
    # can massage distribution configuration to effect compressibility
    fw = firm.distribution(proc.compiler,bins=emu_cfg['M'],M=emu_cfg['M'])
    proc.config(fw)
    return proc

def summary(emu_cfg, **kwargs):
    proc = emulatedHw(**emu_cfg)
    proc.fu.vrf=list(np.concatenate(([0.,float('inf')],list(reversed(range(emu_cfg['FUVRF_SIZE']*emu_cfg['M']-2))))))
    fw = firm.summaryStats(proc.compiler)
    proc.config(fw)
    return proc

def spatial_sparsity(emu_cfg, **kwargs):
    proc = emulatedHw(**emu_cfg)
    proc.fu.vrf=list(np.concatenate(([0.,float('inf')],list(reversed(range(emu_cfg['FUVRF_SIZE']*emu_cfg['M']-2))))))
    fw = firm.spatialSparsity(proc.compiler, emu_cfg['N'])
    proc.config(fw)
    return proc

def norm_check(emu_cfg, limits=None, **kwargs):
    proc = emulatedHw(**emu_cfg)
    # XXX how to set fru filter values appropriately?
    proc.fu.vrf = list(np.linspace(limits[0], limits[1], 64))
    fw = firm.normCheck(proc.compiler, emu_cfg['M'])
    proc.config(fw)
    return proc

# XXX doesn't work
def activation_predictiveness(emu_cfg, **kwargs):
    # should be run on activation data...
    proc = emulatedHw(**emu_cfg)
    fw = firm.activationPredictiveness(proc.compiler)
    proc.config(fw)
    return proc

# missing total invalidity

def main(emu_cfg, proc, indata, num_frames, verbose=None):
    def v_print(*args):
        if verbose is not None:
            print(*args)

    for frame_idx in range(num_frames):
        v_print("Processing frame "+str(frame_idx)+" of "+str(num_frames))
        inframe = indata[frame_idx][0][0][:][:]
        for i in range(1, indata.shape[2]):
            inframe = np.vstack((indata[frame_idx][0][i][:][:], inframe))

        vidx = 0
        frame_stop = inframe.shape[0] - 1
        while vidx < frame_stop:
            for ibidx in range(emu_cfg['IB_DEPTH']):
                proc.push([inframe[vidx][:], False])
                vidx = vidx + 1
                if vidx >= frame_stop:
                    break
            while len(proc.ib.buffer) > 0:
                s = len(proc.ib.buffer)
                log = proc.run(steps = s)
        proc.push([inframe[-1][:], True])
        log = proc.run(steps=10)

    dd = DeltaDecompressor(
        emu_cfg['N'],
        emu_cfg['DATA_WIDTH'],
        emu_cfg['DELTA_SLOTS'],
        emu_cfg['TB_SIZE'])

    v_print(log['tb'][-1][0][:][:])
    #v_print(log['dc'][-1][1])
    decomp_tb = dd.decompress(log['dc'][-1][1], log['tb'][-1][0], log['tb'][-1][1], log['tb'][-1][2])
    #v_print(decomp_tb)
    if emu_cfg['DATA_TYPE'] == 'fixed_point':
        decomp_tb = np.array(encodedIntTofloat(decomp_tb,emu_cfg['DATA_WIDTH']))

    v_print(decomp_tb)
    v_print(decomp_tb.shape)

    tuobj = TestUtils()
    nodata = n_bit_nodata(emu_cfg['DELTA_SLOTS'], emu_cfg['PRECISION'], emu_cfg['INV'])
    v_nodata = np.full((1, emu_cfg['N']), nodata)
    cr = tuobj.compression_ratio(v_nodata, log['tb'][-1][0], decomp_tb)
    v_print(cr)
    return cr

if __name__ == '__main__':
    emu_cfg = {
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
    emu_cfg['PRECISION'] = int(emu_cfg['DATA_WIDTH']/emu_cfg['DELTA_SLOTS'])
    emu_cfg['INV'] = twos_complement_min(emu_cfg['PRECISION'])

    indata = np.load("output_file.npy")

    #proc = distribution(emu_cfg)
    #proc = summary(emu_cfg)
    #proc = spatial_sparsity(emu_cfg)
    #proc = norm_check(emu_cfg, (np.min(indata), np.max(indata)))
    #proc = activation_predictiveness(emu_cfg)

    # XXX setting data type to int
    emu_cfg['DATA_TYPE'] = 'int'
    indata = indata.astype(int)

    #cr = main(emu_cfg, proc, indata, 10, verbose=True)
    #print(cr)
    #exit()

    args = {'limits':(np.min(indata), np.max(indata))}
    with open("compression_experiment.csv", 'w') as file:
        file.write('firmware, delta slots, compression ratio\n')
        for test in [distribution, summary, spatial_sparsity, norm_check]:
            for delta_slots in [2, 4, 8, 16]:
                emu_cfg['DELTA_SLOTS'] = delta_slots
                emu_cfg['PRECISION'] = int(emu_cfg['DATA_WIDTH']/emu_cfg['DELTA_SLOTS'])
                emu_cfg['INV'] = twos_complement_min(emu_cfg['PRECISION'])

                proc = test(emu_cfg, **args)
                cr = main(emu_cfg, proc, indata, indata.shape[0])
                test_conf = test.__name__ + ", "+ str(delta_slots) + ", " + str(cr)
                file.write(test_conf+'\n')
                print(test_conf)
